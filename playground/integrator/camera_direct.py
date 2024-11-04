from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import PSIntegrator, ADIntegrator, mis_weight

class CameraDirIntegrator(PSIntegrator):

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
            pos__ = mi.Vector2f(pin_attached_local[0], pin_attached_local[1]) * film.size()
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


mi.register_integrator("camera_direct", lambda props: CameraDirIntegrator(props))
