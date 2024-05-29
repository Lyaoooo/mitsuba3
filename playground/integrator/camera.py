from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import PSIntegrator, ADIntegrator, mis_weight

class CameraIntegrator(PSIntegrator):

    def __init__(self, props):
        super().__init__(props)

        # Override the max depth to 2 since this is a direct illumination
        # integrator
        self.max_depth = 2
        self.rr_depth = 2
        # Direct illumination integrators don't need radiative backpropagation
        self.radiative_backprop = False

        # Specify the seed ray generation strategy
        self.project_seed = props.get('project_seed', 'both')
        if self.project_seed not in ['both', 'bsdf', 'emitter']:
            raise Exception(f"Project seed must be one of 'both', 'bsdf', "
                            f"'emitter', got '{self.project_seed}'")


    def render_forward(self,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        shape = (film.crop_size()[1],
                 film.crop_size()[0],
                 film.base_channels_count() + len(self.aov_names()))
        result_grad = dr.zeros(mi.TensorXf, shape=shape)

        sampler_spp = sensor.sampler().sample_count()
        sppc = self.override_spp(self.sppc, spp, sampler_spp)
        sppp = self.override_spp(self.sppp, spp, sampler_spp)
        sppi = self.override_spp(self.sppi, spp, sampler_spp)

        # Discontinuous derivative (and the non-RB continuous derivative)
        if sppp > 0 or sppi > 0 or \
           (sppc > 0 and not self.radiative_backprop):

            # Compute an image with all derivatives attached
            ad_img = self.render_ad(scene, sensor, seed, spp, dr.ADMode.Forward)

            # We should only complain about the parameters not being attached
            # if `ad_img` isn't attached and we haven't used RB for the
            # continuous derivatives.
            if dr.grad_enabled(ad_img) or not self.radiative_backprop:
                dr.forward_to(ad_img)
                grad_img = dr.grad(ad_img)
                result_grad += grad_img

        return result_grad

    def sample_rays(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
    ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f, mi.Float]:
        """
        Sample a 2D grid of primary rays for a given sensor

        Returns a tuple containing

        - the set of sampled rays
        - a ray weight (usually 1 if the sensor's response function is sampled
          perfectly)
        - the continuous 2D image-space positions associated with each ray
        """

        film = sensor.film()
        film_size = film.crop_size()
        rfilter = film.rfilter()
        border_size = rfilter.border_size()

        if film.sample_border():
            film_size += 2 * border_size

        spp = sampler.sample_count()

        # Compute discrete sample position
        idx = dr.arange(mi.UInt32, dr.prod(film_size) * spp)

        # Try to avoid a division by an unknown constant if we can help it
        log_spp = dr.log2i(spp)
        if 1 << log_spp == spp:
            idx >>= dr.opaque(mi.UInt32, log_spp)
        else:
            idx //= dr.opaque(mi.UInt32, spp)

        # Compute the position on the image plane
        pos = mi.Vector2i()
        pos.y = idx // film_size[0]
        pos.x = dr.fma(-film_size[0], pos.y, idx)

        if film.sample_border():
            pos -= border_size

        pos += mi.Vector2i(film.crop_offset())

        # Cast to floating point and add random offset
        pos_f = mi.Vector2f(pos) + sampler.next_2d()

        # Re-scale the position to [0, 1]^2
        scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
        offset = -mi.ScalarVector2f(film.crop_offset()) * scale
        pos_adjusted = dr.fma(pos_f, scale, offset)

        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        wavelength_sample = 0
        if mi.is_spectral:
            wavelength_sample = sampler.next_1d()

        ray, weight = sensor.sample_ray(
            time=time,
            sample1=wavelength_sample,
            sample2=pos_adjusted,
            sample3=aperture_sample
        )

        # With box filter, ignore random offset to prevent numerical instabilities
        splatting_pos = mi.Vector2f(pos) if rfilter.is_box_filter() else pos_f

        return ray, weight, splatting_pos


    def render_ad(self,
                  scene: mi.Scene,
                  sensor: Union[int, mi.Sensor],
                  seed: int,
                  spp: int,
                  mode: dr.ADMode) -> mi.TensorXf:
        """
        Renders and accumulates the outputs of the primarily visible
        discontinuities, indirect discontinuities and continuous derivatives.
        It outputs an attached tensor which should subsequently be traversed by
        a call to `dr.forward`/`dr.backward`/`dr.enqueue`/`dr.traverse`.

        Note: The continuous derivatives are only attached if
        `radiative_backprop` is `False`. When using RB for the continuous
        derivatives it should be manually added to the gradient obtained by
        traversing the result of this method.
        """
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        aovs = self.aov_names()
        shape = (film.crop_size()[1],
                 film.crop_size()[0],
                 film.base_channels_count() + len(aovs))
        result_img = dr.zeros(mi.TensorXf, shape=shape)

        sampler_spp = sensor.sampler().sample_count()
        sppc = self.override_spp(self.sppc, spp, sampler_spp)
        sppp = self.override_spp(self.sppp, spp, sampler_spp)
        sppi = self.override_spp(self.sppi, spp, sampler_spp)

        silhouette_shapes = scene.silhouette_shapes()
        has_silhouettes = len(silhouette_shapes) > 0


        # This isn't serious, so let's just warn once
        if has_silhouettes and not film.sample_border() and self.sample_border_warning:
            self.sample_border_warning = False
            mi.Log(mi.LogLevel.Warn,
                "PSIntegrator detected the potential for image-space "
                "motion due to differentiable shape parameters. To correctly "
                "account for shapes entering or leaving the viewport, it is "
                "recommended that you set the film's 'sample_border' parameter "
                "to True.")

        # # Primarily visible discontinuous derivative
        # if sppp > 0 and has_silhouettes:
        #     with dr.suspend_grad():
        #         self.proj_detail.init_primarily_visible_silhouette(scene, sensor)

        #     sampler, spp = self.prepare(sensor, 0xffffffff ^ seed, sppp, aovs)
        #     result_img += self.render_primarily_visible_silhouette(scene, sensor, sampler, spp)


        # Continuous derivative (only if radiative backpropagation is not used)
        # if sppc > 0 and (not self.radiative_backprop):
        with dr.suspend_grad():
            sampler, spp = self.prepare(sensor, seed, sppc, aovs)
            ray, weight, pos = self.sample_rays(scene, sensor, sampler)

            # Launch the Monte Carlo sampling process in differentiable mode
            L, valid, aovs, _, first_it_t = self.sample(
                mode     = dr.ADMode.Primal,
                # mode     = mode,
                scene    = scene,
                sampler  = sampler,
                ray      = ray,
                depth    = 0,
                δL       = None,
                state_in = None,
                active   = mi.Bool(True),
                project  = False,
                si_shade = None
            )

        # To support camera intrinsics derivatives
        if True:
            with dr.suspend_grad():
                film = sensor.film()
                camera_to_sample = mi.perspective_projection(
                    film.size(),
                    film.crop_size(),
                    film.crop_offset(),
                    mi.traverse(sensor)["x_fov"][0],
                    sensor.near_clip(),
                    sensor.far_clip()
                )

                if True:
                    # Min distance filter
                    if False:
                        # reduce op (no cuda support)
                        pos_i = mi.Vector2i(pos)
                        pixel_index = pos_i.y * film.size()[0] + pos_i.x
                        res = dr.zeros(mi.Float, film.size()[0] * film.size()[1])
                        dr.scatter_reduce(dr.ReduceOp.Min, res, first_it_t, pixel_index, valid)
                        distance = dr.gather(mi.Float, res, pixel_index, valid)
                    else:
                        # loop version
                        pos_i = mi.Vector2i(pos)
                        pixel_index = pos_i.y * film.size()[0] + pos_i.x

                        num_pixels = film.size()[0] * film.size()[1]
                        res = dr.full(mi.Float, dr.inf, num_pixels)
                        idx = dr.arange(mi.UInt32, num_pixels) * spp
                        first_it_t[~valid] = dr.inf
                        iter = mi.UInt32(0)

                        loop = mi.Loop("Loop scatter reduce", lambda: (iter, res))
                        while loop(iter < spp):
                            t_tmp = dr.gather(mi.Float, first_it_t, idx + iter)
                            res = dr.select(t_tmp < res, t_tmp, res)
                            iter += 1

                        res[dr.isinf(res)] = 1
                        distance = dr.gather(mi.Float, res, pixel_index)
                else:
                    distance = first_it_t

            to_world = sensor.world_transform()
            to_local = to_world.inverse()
            world_to_sample = camera_to_sample @ to_local

            pin_attached_local = world_to_sample @ dr.detach(ray(distance))
            pos__ = mi.Vector2f(pin_attached_local[0], pin_attached_local[1]) * film.size()  # Why * film.size()?
            pos = dr.replace_grad(pos, pos__)

        block = film.create_block()
        block.set_coalesce(block.coalesce() and sppc >= 4)

        ADIntegrator._splat_to_block(
            block, film, pos,
            value=L * weight,
            weight=1.0,
            alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
            aovs=[],
            wavelengths=ray.wavelengths
        )

        film.put_block(block)
        result_img += film.develop()

        return result_img


    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               depth: mi.UInt32,
               δL: Optional[mi.Spectrum],
               state_in: Any,
               active: mi.Bool,
               project: bool = False,
               si_shade: Optional[mi.SurfaceInteraction3f] = None,
               **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum, mi.Bool, List[mi.Float], Any]:
        """
        See ``PSIntegrator.sample()`` for a description of this interface and
        the role of the various parameters and return values.
        """
        del depth, δL, state_in, kwargs  # Unused


        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = mode == dr.ADMode.Primal

        # Should we use ``si_shade`` as the first interaction and ignore the ray?
        ignore_ray = si_shade is not None

        # Standard BSDF evaluation context
        bsdf_ctx = mi.BSDFContext()

        L = mi.Spectrum(0)

        # ---------------------- Direct emission ----------------------

        # Use `si_shade` as the first interaction or trace a ray to the first
        # interaction
        if ignore_ray:
            si = si_shade
        else:
            with dr.resume_grad(when=not primal):
                si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All,
                             coherent=True, active=active)
        first_it_t = dr.detach(mi.Float(si.t))

        # Hide the environment emitter if necessary
        if not self.hide_emitters:
            with dr.resume_grad(when=not primal):
                L += si.emitter(scene).eval(si, active)

        active_next = active & si.is_valid() & (self.max_depth > 1)

        # Get the BSDF
        bsdf = si.bsdf(ray)

        # ---------------------- Emitter sampling ----------------------

        # Is emitter sampling possible on the current vertex?
        active_em_ = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

        # If so, pick an emitter and sample a detached emitter direction
        ds_em, emitter_val = scene.sample_emitter_direction(
            si, sampler.next_2d(active_em_), test_visibility=True, active=active_em_)
        active_em = active_em_ & dr.neq(ds_em.pdf, 0.0)

        with dr.resume_grad(when=not primal):
            # Evaluate the BSDF (foreshortening term included)
            wo = si.to_local(ds_em.d)
            bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)

            # Re-compute some values with AD attached only in differentiable
            # phase
            if not primal:
                # Re-compute attached `emitter_val` to enable emitter optimization
                ds_em.d = dr.normalize(ds_em.p - si.p)
                spec_em = scene.eval_emitter_direction(si, ds_em, active_em)
                emitter_val = spec_em / ds_em.pdf
                dr.disable_grad(ds_em.d)

            # Compute the detached MIS weight for the emitter sample
            mis_em = dr.select(ds_em.delta, 1.0, mis_weight(ds_em.pdf, bsdf_pdf))

            L[active_em] += bsdf_val * emitter_val * mis_em

        # ---------------------- BSDF sampling ----------------------

        # Perform detached BSDF sampling
        sample_bsdf, weight_bsdf = bsdf.sample(bsdf_ctx, si, sampler.next_1d(active_next),
                                               sampler.next_2d(active_next), active_next)
        active_bsdf = active_next & dr.any(dr.neq(weight_bsdf, 0.0))
        delta_bsdf = mi.has_flag(sample_bsdf.sampled_type, mi.BSDFFlags.Delta)

        # Construct the BSDF sampled ray
        ray_bsdf = si.spawn_ray(si.to_world(sample_bsdf.wo))

        with dr.resume_grad(when=not primal):
            # Re-compute `weight_bsdf` with AD attached only in differentiable
            # phase
            if not primal:
                wo = si.to_local(ray_bsdf.d)
                bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active_bsdf)
                weight_bsdf = bsdf_val / dr.detach(bsdf_pdf)

                # ``ray_bsdf`` is left detached (both origin and direction)

            # Trace the BSDF sampled ray
            si_bsdf = scene.ray_intersect(
                ray_bsdf, ray_flags=mi.RayFlags.All, coherent=False, active=active_bsdf)

            # Evaluate the emitter
            L_bsdf = si_bsdf.emitter(scene, active_bsdf).eval(si_bsdf, active_bsdf)

            # Compute the detached MIS weight for the BSDF sample
            ds_bsdf = mi.DirectionSample3f(scene, si=si_bsdf, ref=si)
            pdf_emitter = scene.pdf_emitter_direction(
                ref=si, ds=ds_bsdf, active=active_bsdf & ~delta_bsdf)
            mis_bsdf = dr.select(delta_bsdf, 1.0, mis_weight(sample_bsdf.pdf, pdf_emitter))

            L[active_bsdf] += L_bsdf * weight_bsdf * mis_bsdf

        # ---------------------- Seed rays for projection ----------------------

        guide_seed = []
        if project:
            if self.project_seed == "bsdf":
                # A BSDF sample can only be a valid seed ray if it intersects a
                # shape
                active_guide = active_bsdf & si_bsdf.is_valid()

                guide_seed = [dr.detach(ray_bsdf), mi.Bool(active_guide)]
            elif self.project_seed == "emitter":
                # Return emitter sampling rays including the ones that fail
                # `test_visibility`
                ray_em = si.spawn_ray_to(ds_em.p)
                ray_em.maxt = dr.largest(mi.Float)

                # Directions towards the interior have no contribution for
                # direct integrators
                active_guide = active_em_ & (dr.dot(si.n, ray_em.d) > 0)

                guide_seed = [dr.detach(ray_em), mi.Bool(active_guide)]
            elif self.project_seed == "both":
                # By default we use the emitter sample as the seed ray
                ray_seed = si.spawn_ray_to(ds_em.p)
                ray_seed.maxt = dr.largest(mi.Float)
                active_guide = active_em_ & (dr.dot(si.n, ray_seed.d) > 0)

                # Flip a coin only when both samples are valid
                mask_replace = (active_bsdf & si_bsdf.is_valid()) & \
                               ((sampler.next_1d() > 0.5) | ~active_guide)
                ray_seed[mask_replace] = ray_bsdf

                guide_seed = [dr.detach(ray_seed), active_guide | mask_replace]

        return L, active, [], guide_seed if project else None, first_it_t


    # def sample_radiance_difference(self, scene, ss, curr_depth, sampler, active):
    #     if curr_depth == 1:

    #         # ----------- Estimate the radiance of the background -----------

    #         ray_bg = ss.spawn_ray()
    #         si_bg = scene.ray_intersect(ray_bg, active=active)
    #         radiance_bg = si_bg.emitter(scene).eval(si_bg, active)

    #         # ----------- Estimate the radiance of the foreground -----------

    #         # For direct illumination integrators, only an area emitter can
    #         # contribute here. It is possible to call ``sample()`` to estimate
    #         # this contribution. But to avoid the overhead we simply query the
    #         # emitter here to obtain the radiance.
    #         si_fg = dr.zeros(mi.SurfaceInteraction3f)

    #         # We know the incident direction is valid since this is the
    #         # foreground interaction. Overwrite the incident direction to avoid
    #         # potential issues introduced by smooth normals.
    #         si_fg.wi = mi.Vector3f(0, 0, 1)
    #         radiance_fg = ss.shape.emitter().eval(si_fg, active)
    #     elif curr_depth == 0:

    #         # ----------- Estimate the radiance of the background -----------
    #         ray_bg = ss.spawn_ray()
    #         radiance_bg, _, _, _ = self.sample(
    #             dr.ADMode.Primal, scene, sampler, ray_bg, curr_depth, None, None, active, False, None)

    #         # ----------- Estimate the radiance of the foreground -----------
    #         # Create a preliminary intersection point
    #         pi_fg = dr.zeros(mi.PreliminaryIntersection3f)
    #         pi_fg.t = 1
    #         pi_fg.prim_index = ss.prim_index
    #         pi_fg.prim_uv = ss.uv
    #         pi_fg.shape = ss.shape

    #         # Create a dummy ray that we never perform ray-intersection with to
    #         # compute other fields in ``si``
    #         dummy_ray = mi.Ray3f(ss.p - ss.d, ss.d)

    #         # The ray origin is wrong, but this is fine if we only need the primal
    #         # radiance
    #         si_fg = pi_fg.compute_surface_interaction(
    #             dummy_ray, mi.RayFlags.All, active)

    #         # If smooth normals are used, it is possible that the computed
    #         # shading normal near visibility silhouette points to the wrong side
    #         # of the surface. We fix this by clamping the shading frame normal
    #         # to the visible side.
    #         alpha = dr.dot(si_fg.sh_frame.n, ss.d)
    #         eps = dr.epsilon(alpha) * 100
    #         wrong_side = active & (alpha > -eps)

    #         # NOTE: In the following case, (1) a single sided BSDF is used,
    #         # (2) the silhouette sample is on an open boundary like an open
    #         # edge, and (3) we actually hit the back side of the surface,
    #         # the expected radiance is zero because no BSDF is defiend on
    #         # that side. But this shading frame correction will mistakenly
    #         # produce a non-zero radiance. Please use two-sided BSDFs if
    #         # this is a concern.

    #         # Remove the component of the shading frame normal that points to
    #         # the wrong side
    #         new_sh_normal = dr.normalize(
    #             si_fg.sh_frame.n - (alpha + eps) * ss.d)
    #         # `si_fg` surgery
    #         si_fg.sh_frame[wrong_side] = mi.Frame3f(new_sh_normal)
    #         si_fg.wi[wrong_side] = si_fg.to_local(-ss.d)

    #         # Estimate the radiance starting from the surface interaction
    #         radiance_fg, _, _, _ = self.sample(
    #             dr.ADMode.Primal, scene, sampler, ray_bg, curr_depth, None, None, active, False, si_fg)

    #     else:
    #         raise Exception(f"Unexpected depth {curr_depth} in direct projective integrator")

    #     # Compute the radiance difference
    #     radiance_diff = radiance_fg - radiance_bg
    #     active_diff = active & (dr.max(dr.abs(radiance_diff)) > 0)

    #     return radiance_diff, active_diff




    # def render_primarily_visible_silhouette(self,
    #                                         scene: mi.Scene,
    #                                         sensor: mi.Sensor,
    #                                         sampler: mi.Sampler,
    #                                         spp: int) -> mi.TensorXf:
    #     """
    #     Renders the primarily visible discontinuities.

    #     This method returns the AD-attached image. The result must still be
    #     traversed using one of the Dr.Jit functions to propagate gradients.
    #     """
    #     film = sensor.film()
    #     aovs = self.aov_names()

    #     # Explicit sampling to handle the primarily visible discontinuous derivative
    #     with dr.suspend_grad():
    #         # Get the viewpoint
    #         sensor_center = sensor.world_transform() @ mi.Point3f(0)

    #         # Sample silhouette point
    #         ss = self.proj_detail.sample_primarily_visible_silhouette(
    #             scene, sensor_center, sampler.next_2d(), True)
    #         active = ss.is_valid() & (ss.pdf > 0)

    #         # Jacobian (motion correction included)
    #         J = self.proj_detail.perspective_sensor_jacobian(sensor, ss)

    #         ΔL = self.proj_detail.eval_primary_silhouette_radiance_difference(
    #             scene, sampler, ss, sensor_center, active=active)
    #         active &= dr.any(dr.neq(ΔL, 0))

    #     # ∂z/∂ⲡ * normal
    #     si = dr.zeros(mi.SurfaceInteraction3f)
    #     si.p = ss.p
    #     si.prim_index = ss.prim_index
    #     si.uv = ss.uv
    #     p = ss.shape.differential_motion(dr.detach(si), active)

    #     p = sensor.world_transform() @ p
    #     p = -1 * p
    #     # print(sensor.world_transform())
    #     # print(dr.grad(sensor.world_transform()))
        
    #     motion = dr.dot(p, ss.n)

    #     # Compute the derivative (motion)
    #     derivative = ΔL * motion * dr.rcp(ss.pdf) * J

    #     # Prepare a new imageblock and compute splatting coordinates
    #     film.prepare(aovs)
    #     with dr.suspend_grad():
    #         it = dr.zeros(mi.Interaction3f)
    #         it.p = ss.p
    #         sensor_ds, _ = sensor.sample_direction(it, mi.Point2f(0))

    #     # Particle tracer style imageblock to accumulate primarily visible derivatives
    #     block = film.create_block(normalize=True)
    #     block.set_coalesce(block.coalesce() and spp >= 4)
    #     block.put(
    #         pos=sensor_ds.uv,
    #         wavelengths=[],
    #         value=derivative * dr.rcp(mi.ScalarFloat(spp)),
    #         weight=0,
    #         alpha=1,
    #         active=active
    #     )
    #     film.put_block(block)

    #     return film.develop()
    

mi.register_integrator("camera", lambda props: CameraIntegrator(props))
