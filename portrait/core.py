import jax
import jax.numpy as jnp
import numpy as np


def coverage(time, gap=0.1, precision=1e-3):
    """Returns the phase coverage for given period(s)

    Parameters
    ----------
    times : list of arrays
        an array of observed times
    gap : float, optional
        minimum gap between observations to be considered an independent
        segment, by default 0.5
    precision : float, optional
        precision of the phase coverage returned, by default 1e-3

    Returns
    -------
    function
        a function that computes the phase coverage for a given period,
        i.e. with signature :code:`fun(float or array) -> float or array`
    """
    # we pre-compute segments_times: the pairs of (time_min, time_max) of each segment
    diff_time = jnp.diff(time)
    segment_idxs = jnp.flatnonzero(diff_time > gap)
    segment_idxs = jnp.sort(
        jnp.hstack([0, *segment_idxs, *segment_idxs + 1, len(time) - 1])
    )
    cuts_time = time[segment_idxs]
    segments_times = cuts_time.reshape(-1, 2)

    # a grid of well sampled phases
    sampled = jnp.arange(0.0, 1.0, precision)

    @jax.jit
    def fun(period):
        """Compute the phase coverage for a given period

        Parameters:
        ----------
        period: float or array
            the period(s) to compute the phase coverage for

        Returns
        -------
        float or array
            the phase coverage for the given period(s)
        """

        raw_segments_phases = ((segments_times + 0.5 * period) % period) / period

        # segments_phases is unordered, some of them are
        # (0.6, 0.5) which corresponds to a segment that wraps around
        # the phase 1.0. We need to fix this and split it in (0.0, 0.5) and (0.6, 1.0)
        # we allocate segments_phases_2 for the extra split, as JAX requires fixed size arrays

        n = raw_segments_phases.shape[0]

        is_positive = jnp.array(raw_segments_phases[:, 1] >= raw_segments_phases[:, 0])

        segments_phases_1 = jnp.where(
            is_positive,
            raw_segments_phases.T,
            jnp.vstack([jnp.zeros(n), raw_segments_phases[:, 1]]),
        ).T

        segments_phases_2 = jnp.where(
            is_positive,
            jnp.zeros_like(raw_segments_phases).T,
            jnp.vstack([raw_segments_phases[:, 0], jnp.ones(n)]),
        ).T

        # if the phase covers the whole period, we need to add a segment of full phase,
        # i.e. (0.0, 1.0), because the bounds are meaningless

        is_complete = jnp.any(
            jnp.array(segments_times[:, 1] - segments_times[:, 0] >= period)
        )

        segments_complete = jnp.where(
            is_complete,
            jnp.array([0.0, 1.0]),
            jnp.array([0.0, 0.0]),
        ).T

        # we now have clean segments from which to compute the overlap
        # on the grid of sampled phases
        clean_segments_phases = jnp.vstack(
            [segments_phases_1, segments_phases_2, segments_complete]
        )

        overlap = jnp.array(
            (sampled[:, None] >= clean_segments_phases[:, 0])
            & (sampled[:, None] <= clean_segments_phases[:, 1])
        ).astype(float)

        return jnp.mean(jnp.max(overlap, 1))

    return jnp.vectorize(fun)


def phase(time, t0, period):
    time = np.asarray(time)
    phases = (time[..., None] - t0 + 0.5 * period) % period - 0.5 * period
    if np.isscalar(period):
        return phases.T[0]
    else:
        return phases


def period_match(t0s, periods, tolerance=0.001):
    """Returns a periodogram with period matching most input times

    Parameters
    ----------
    t0s : array
        list of event observed times
    periods : float or array
        periods to match
    tolerance : float, optional
        timing error, by default 0.001

    Returns
    -------
    tuple(np.array, float)
        - number of event matched per period
        - best period
    """
    match = np.count_nonzero(np.abs(phase(t0s, t0s[0], periods)) < tolerance, 0)
    best = periods[np.flatnonzero(match == np.max(match))[-1]]
    return match, best
