import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# SOME UTILS...

# TODO: reimplement it more fancy...
def discard_nans(array):
    new = []
    for el in array:
        if not np.isnan(el):
            new.append(el)
    return np.asarray(new)

def rad2deg(rad):
    return rad*360/(2*np.pi)

def deg2rad(deg):
    return deg*(2*np.pi)/360.

def circmedian(angs, unit='rad'):
    # from https://github.com/scipy/scipy/issues/6644
    # Radians!
    pdists = angs[np.newaxis, :] - angs[:, np.newaxis]
    if unit == 'rad':
        pdists = (pdists + np.pi) % (2 * np.pi) - np.pi
    elif unit == 'deg':
        pdists = (pdists +180) % (360.) - 180.
    pdists = np.abs(pdists).sum(1)

    # If angs is odd, take the center value
    if len(angs) % 2 != 0:
        return angs[np.argmin(pdists)]
    # If even, take the mean between the two minimum values
    else:
        index_of_min = np.argmin(pdists)
        min1 = angs[index_of_min]
        # Remove minimum element from array and recompute
        new_pdists = np.delete(pdists, index_of_min)
        new_angs = np.delete(angs, index_of_min)
        min2 = new_angs[np.argmin(new_pdists)]
        if unit == 'rad':
            return scipy.stats.circmean([min1, min2], high=np.pi, low=-np.pi)
        elif unit == 'deg':
            return scipy.stats.circmean([min1, min2], high=180., low=-180.)




def ambisonic_coefs(azi, ele):
    """
    Get ambisonic coefficients for the given direction
    :param azi: in rad!
    :param ele: in rad!
    :return: ambi coefs in acn, n3d
    """
    w = 1.
    y = np.sqrt(3) * np.sin(azi) * np.cos(ele)
    z = np.sqrt(3) * np.sin(ele)
    x = np.sqrt(3) * np.cos(azi) * np.cos(ele)
    return np.asarray([w, y, z, x])


# TODO: FIND MAX-RE COEFS FOR FIRST ORDER
def beamforming(b_format, azi, ele, method='basic'):
    """
    :param b_format: FOA signal in acn, n3d. Shape must be [num_frames, num_channels] (as given by soundfile.read)
    :param azi: beamformer azimuth direction in degree.
    :param ele: beamformer elevation direction in degree.
    :param method: 'basic', 'inphase' or 'maxre'. Default to 'basic.
    :return: mono source
    """
    M = 4 # number of ambisonic channels

    # Assert b_format dimensions
    assert b_format.ndim == 2
    assert b_format.shape[1] == M

    # Get beam directiviyty gains
    if method == 'basic':
        alpha = np.ones(4)
    elif method == 'inphase':
        alpha = np.asarray([1., 1/3., 1/3., 1/3.])
    elif method == 'maxre':
        raise NotImplementedError
    else:
        raise ValueError

    num_frames = b_format.shape[0]
    out = np.zeros(num_frames)

    # Get beamformer
    ambi_gains = ambisonic_coefs(deg2rad(azi), deg2rad(ele))
    for m in range(M):
        out += b_format[:, m] * ambi_gains[m] * alpha[m]
    out = out / float(M)  # Don't forget to divide by the number of ambisonic channels!!!

    return out




class HybridKMeans:
    def __init__(self, k=2, delta=1e-6, max_iter = 100, n_init=10, plot=False):
        self.k_ = k
        self.delta_ = delta
        self.max_iter_ = max_iter
        self.n_init_ = n_init

        self.cluster_centers_ = None
        self.closest_centers_ = None
        self.plot_ = plot

    def fit(self, X):

        # if k=1, just compute the mean...
        if self.k_ == 1:
            kmeans = HybridKmeans_implementation(self.k_, self.delta_, self.max_iter_).run(X)
            if self.plot_:
                kmeans.plot(X, title='0')
            return kmeans
        else:
            kmeans_list = []
            inertias_list = []
            for n in range(self.n_init_):
                kmeans = HybridKmeans_implementation(self.k_, self.delta_, self.max_iter_).run(X)
                kmeans_list.append(kmeans)
                inertias_list.append(kmeans.inertia_)
                if self.plot_:
                    kmeans.plot(X, title=str(n))
            return kmeans_list[int(np.argmin(np.asarray(inertias_list)))]



class HybridKmeans_implementation:

    def __init__(self, k=2, delta=1e-6, max_iter = 100):
        self.k_ = k
        self.delta_ = delta
        self.max_iter_ = max_iter

        self.cluster_centers_ = None
        self.closest_centers_ = None
        self.inertia_ = self.delta_
        self.iter_ = 0

    def compute_spherical_distance(self, p1, p2):
        """
        Compute great-circle distance around the sphere by the spherical law of cosines
        For more info, check https://en.wikipedia.org/wiki/Great-circle_distance

        :param p1: [azi, ele] in radians
        :param p2: [azi, ele] in radians
        :return: spherical distance
        """
        #TODO: HOLD WARNINGS FOR THE CASE OF K*PI/2
        d = np.arccos((np.sin(p1[1]) * np.sin(p2[1])) + (np.cos(p1[1]) * np.cos(p2[1]) * np.cos(p2[0] - p1[0])))
        return d

    def initialize_centroids(self, points):
        """returns k centroids from the initial points"""
        centroids = points.copy()
        np.random.shuffle(centroids)
        self.cluster_centers_ = centroids[:self.k_]

    def closest_centroid(self, points):
        """returns an array containing the index to the nearest centroid for each point"""

        distances = np.array([[self.compute_spherical_distance(p, c) for c in self.cluster_centers_] for p in points ])
        self.closest_centers_ = np.argmin(distances, axis=1)

        closest_distances = np.zeros(len(points))
        for d_idx, d in enumerate(distances.T):
            closest_distances[d_idx] = d[self.closest_centers_[d_idx]]

        self.inertia_ = np.sum(closest_distances)


    def move_centroids(self, points):
        """returns the new centroids assigned from the points closest to them"""

        centers = np.zeros(self.cluster_centers_.shape)
        for k in range(self.cluster_centers_.shape[0]):
            points_in_cluster = points[self.closest_centers_ == k]

            # TODO: CHANGE PERIODICITY HERE
            azi_mean = scipy.stats.circmean(points_in_cluster[:,0], high=np.pi, low=-np.pi)
            ele_mean = np.mean(points_in_cluster[:,1])
            centers[k] = [azi_mean,ele_mean]

        self.cluster_centers_ = centers


    def plot(self, points, title=None):
        plt.figure()
        plt.suptitle(title)
        plt.scatter(points[:, 0], points[:, 1], c=self.closest_centers_)
        for c in self.cluster_centers_:
            plt.scatter(c[0], c[1], c='r')
            plt.xlim(-np.pi, np.pi)
            plt.ylim(-np.pi/2, np.pi/2)
        plt.show()


    def run(self, X):
        last_inertia = self.inertia_*3 # Whatever number to ensure first entry in the while loop
        self.initialize_centroids(X)
        while np.abs(last_inertia - self.inertia_) > self.delta_:
            if self.iter_ > self.max_iter_:
                break
            last_inertia = self.inertia_
            self.closest_centroid(X)
            self.move_centroids(X)
            # self.plot(X, title=str(self.iter_))
            self.iter_ += 1
        return self
