import numpy

from numpy.typing import NDArray

from dask.delayed import delayed

rng = numpy.random.default_rng()


# TODO: Add velocities
# TODO: Add possibility to give velocity offsets
# WORK IN PROGRESS
@delayed
def sky_positions(
    n_sky_points_per_cluster: int,
    n_clusters: int = 1,
    cluster_shape: str = "sphere",
    distribution_str: str = "uniform",
    cluster_sizes: list[float] | NDArray[numpy.float64] = [1],
    velocities: bool = True,  # TODO da aggiungere
    offset_vector: numpy.ndarray = numpy.array([0, 0, 0]),
    on_the_surface: bool = False,
):
    # DOCUMENT THIS !!!
    supported_cluster_shape = [
        "sphere",
        "disk",
        "cube",
        "square",
        "ellipse",
        "ellipsoid",
        "cuboid",
        "rectangle",
    ]

    supported_distributions_str = [
        "norm",
        "uniform",
    ]

    # Checking input
    if (not isinstance(n_sky_points_per_cluster, int)) or (
        not isinstance(n_clusters, int)
    ):
        raise ValueError("Number of sources and clusters must be integers")
    if cluster_shape not in supported_cluster_shape:
        raise ValueError(
            f"Unsupported shape for cluster\nOptions are {supported_cluster_shape}"
        )
    if (cluster_shape in ["sphere", "disk", "square", "cube"]) and (
        len(cluster_sizes) < 1
    ):
        raise ValueError(
            f"If shape is {cluster_shape}, you must provide at least 1 value for cluster_size\nOnly the first will be used"
        )
    if (cluster_shape in ["ellipse", "rectangle"]) and (len(cluster_sizes) < 2):
        raise ValueError(
            f"If shape is {cluster_shape}, you must provide at least 2 values for cluster_size\nOnly the first 2 will be used"
        )
    if (cluster_shape in ["ellipsoid", "cuboid"]) and (len(cluster_sizes) < 3):
        raise ValueError(
            f"If shape is {cluster_shape}, you must provide 3 values for cluster_siz\nOnly the first 3 will be usede"
        )
    if distribution_str not in supported_distributions_str:
        raise ValueError(
            f"Unsupported distribution_str\nOptions are {supported_distributions_str}"
        )

    cube_norm = (cluster_shape in ["square", "cube", "cuboid", "rectangle"]) and (
        distribution_str == "norm"
    )

    sky_points = numpy.zeros([n_clusters, n_sky_points_per_cluster])

    if (cluster_shape in ["sphere", "disk", "ellipse", "ellipsoid"]) or cube_norm:
        # First generate a sphere with the right distribution
        if (distribution_str == "uniform") or on_the_surface:
            radius_cube = rng.uniform(0, 1, [n_clusters, n_sky_points_per_cluster])
            radius = numpy.power(radius_cube, 1 / 3)

            phi = rng.uniform(0, 2 * numpy.pi, [n_clusters, n_sky_points_per_cluster])
            costheta = rng.uniform(-1, 1, [n_clusters, n_sky_points_per_cluster])
            theta = numpy.arccos(costheta)

            if on_the_surface:
                radius = 1

            x = radius * numpy.sin(theta) * numpy.cos(phi)
            y = radius * numpy.sin(theta) * numpy.sin(phi)
            z = radius * costheta

            non_shaped_sky_points = numpy.array([x, y, z])
            non_scaled_sky_points = non_shaped_sky_points  # / numpy.nanmax(numpy.abs(non_shaped_sky_points))
            # Normalizing points between -1 and 1
            scaled_sky_points = non_scaled_sky_points * cluster_sizes[0] / 2
            sky_points = numpy.einsum("ijk->jki", scaled_sky_points)

        elif distribution_str == "norm":
            # Making a simmetric 3D multivariate norm distribution
            sky_points = rng.multivariate_normal(
                [0, 0, 0],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [n_clusters, n_sky_points_per_cluster],
            )
            # scaling between -1 and 1
            norm_sky_points = sky_points / numpy.abs(sky_points).max()
            # scaling to have the right shape
            sky_points = norm_sky_points * cluster_sizes[0] / 2

    elif cluster_shape in ["cube", "square", "rectangle", "cuboid"]:
        if distribution_str == "uniform":
            sky_points = rng.uniform(
                -cluster_sizes[0] / 2,
                cluster_sizes[0] / 2,
                [n_clusters, n_sky_points_per_cluster, 3],
            )

            if on_the_surface:
                cube_faces = numpy.array(
                    [
                        [[0, 1, 1], [1, 0, 0]],
                        [[0, 1, 1], [-1, 0, 0]],
                        [[1, 0, 1], [0, 1, 0]],
                        [[1, 0, 1], [0, -1, 0]],
                        [[1, 1, 0], [0, 0, 1]],
                        [[1, 1, 0], [0, 0, -1]],
                    ]
                )
                reshaped_cube_faces = numpy.einsum("ijk -> ikj", cube_faces)
                mask = rng.choice(
                    reshaped_cube_faces, [n_clusters, n_sky_points_per_cluster]
                )
                mask_prod = mask[:, :, :, 0]
                mask_sum = mask[:, :, :, 1] * cluster_sizes[0] / 2
                sky_points = numpy.einsum("ijk, ijk -> ijk", sky_points, mask_prod)
                sky_points = sky_points + mask_sum

    if cluster_shape in ["ellipsoid", "cuboid", "ellipse", "rectangle"]:
        y = sky_points[:, :, 1]
        sky_points[:, :, 1] = y * (cluster_sizes[1] / cluster_sizes[0])
        if cluster_shape in ["ellipsoid", "cuboid"]:
            #! NON FUNZIONA PER LE ELLISSOIDI
            # NOTE: Quando si considerano le ellissoidi, per qualche motivo non
            # NOTE: non funziona il fatto di stretchare la sfera. Devo capire meglio
            z = sky_points[:, :, 2]
            sky_points[:, :, 2] = z * (cluster_sizes[2] / cluster_sizes[0])

    if cluster_shape in ["disk", "ellipse", "square", "rectangle"]:
        sky_points[:, :, 2] = 0

    # Adding the offset
    if (len(offset_vector.shape) == len(sky_points.shape)) and (
        offset_vector.shape[1] == sky_points.shape[0]
    ):
        offsetted_sky_points = sky_points + offset_vector[0][:, None, :]
    else:
        offsetted_sky_points = sky_points + offset_vector

    return offsetted_sky_points


# WORK IN PROGRESS !!!
@delayed
def noise(
    domain: str = "frequency",
    range: list[float] | NDArray[numpy.float64] = [0, 2048],
    distribution: str = "exponential",
    # TODO: METTERE IL VALORE GIUSTO
    params: list[float] | NDArray[numpy.float64] = [1e-20],
    resolution: float = 1,
    n_samples: float = 1,
) -> NDArray:
    # DOCUMENT THIS !!!
    ...
