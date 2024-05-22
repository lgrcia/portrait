import numpy as np


# Much faster than the previous one
def _coverage(times, period):
    if period == 0:
        return 1
    else:
        ph = (times + 0.5 * period) % period - (0.5 * period)
        sph_in = np.sort(ph)
        sph_out = sph_in[::-1]

        sph_in_diff = np.abs(np.diff(sph_in))
        sph_out_diff = np.abs(np.diff(sph_out))

        df = np.min(np.diff(times))

        spaces_in = np.sort(
            sph_in[np.hstack([*np.argwhere(sph_in_diff > 4 * df).T, len(sph_in) - 1])]
        )
        spaces_out = np.sort(
            sph_out[np.hstack([*np.argwhere(sph_out_diff > 4 * df).T, len(sph_in) - 1])]
        )

        return np.sum(spaces_in - spaces_out) / period


vectorize_coverage = np.vectorize(_coverage, signature="(n),()->()")


def coverage(times: list, periods: np.array):
    """Returns the phase coverage for given period(s)

    Parameters
    ----------
    times : list of arrays
        a list of observed times arrays (such as [[0, 1, 2], [5, 6, 7]])
    periods : np.array
        an array of periods (in the same units as :code:`times`)

    Returns
    -------
    np.array
        coverage of each period
    """

    return vectorize_coverage(times, periods)


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
