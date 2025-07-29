import logging
import os
import struct
import time

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

device = cuda.Device(0)
logger.info(
    f"Max threads per block:  {device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)}"
)

kernel_code = """
__host__ __device__ float sqrt7(float x) {
    unsigned int i = *(unsigned int *)&x;
    i += 127 << 23;  // adjust bias
    i >>= 1;         // approximation of square root
    return *(float *)&i;
}

__host__ __device__ float euclidean_distance(float x1, float y1, float z1, float x2, float y2, float z2) {
    return ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
}

/*Calculate the euclidean distance between two 3d points normalized*/
__host__ __device__ float euclidean_distance_norm(float x1, float y1, float z1, float x2, float y2, float z2) {
    return sqrt7((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
}

__device__ int discard_center(float *subject_data, float *atlas_data, unsigned short int ndata_fiber, float threshold, unsigned int fiber_index,
                              unsigned int fatlas_index) {
    unsigned int fpoint = (fiber_index * ndata_fiber) + 31;   // Point of fiber, 31 is the middle of the fiber
    unsigned int apoint = (fatlas_index * ndata_fiber) + 31;  // Atlas point, 31 is the middle of the fiber

    float ed = euclidean_distance(subject_data[fpoint - 1], subject_data[fpoint], subject_data[fpoint + 1], atlas_data[apoint - 1], atlas_data[apoint],
                                  atlas_data[apoint + 1]);
    if (ed > (threshold * threshold))
        return 1;
    else
        return 0;
}

__device__ int discard_extremes(float *subject_data, float *atlas_data, unsigned short int ndata_fiber, float threshold, int *is_inverted,
                                unsigned int fiber_index, unsigned int fatlas_index) {
    unsigned int fpoint1 = fiber_index * ndata_fiber;   // Point 0 of fiber
    unsigned int apoint1 = fatlas_index * ndata_fiber;  // Atlas point 0
    unsigned int fpoint21 = fpoint1 + 62;               // Last point on the fiber
    unsigned int apoint21 = apoint1 + 62;               // Last point on the fiber
    float first_points_ed_direct = euclidean_distance(subject_data[fpoint1], subject_data[fpoint1 + 1], subject_data[fpoint1 + 2], atlas_data[apoint1],
                                                      atlas_data[apoint1 + 1], atlas_data[apoint1 + 2]);
    float first_point_ed_flip = euclidean_distance(subject_data[fpoint1], subject_data[fpoint1 + 1], subject_data[fpoint1 + 2], atlas_data[apoint21 - 2],
                                                   atlas_data[apoint21 - 1], atlas_data[apoint21]);
    float first_points_ed = min(first_points_ed_direct, first_point_ed_flip);

    if (first_points_ed > (threshold * threshold))
        return 1;
    else {
        float last_points_ed;
        if (first_points_ed_direct < first_point_ed_flip) {
            (*is_inverted) = 0;
            last_points_ed = euclidean_distance(subject_data[fpoint21 - 2], subject_data[fpoint21 - 1], subject_data[fpoint21], atlas_data[apoint21 - 2],
                                                atlas_data[apoint21 - 1], atlas_data[apoint21]);
        } else {
            (*is_inverted) = 1;
            last_points_ed = euclidean_distance(subject_data[fpoint21 - 2], subject_data[fpoint21 - 1], subject_data[fpoint21], atlas_data[apoint1],
                                                atlas_data[apoint1 + 1], atlas_data[apoint1 + 2]);
        }
        if (last_points_ed > (threshold * threshold))
            return 1;
        else
            return 0;
    }
}

__device__ int discard_four_points(float *subject_data, float *atlas_data, unsigned short int ndata_fiber, float threshold, int is_inverted,
                                   unsigned int fiber_index, unsigned int fatlas_index) {
    unsigned short int points[4] = {3, 7, 13, 17};
    unsigned short int inv = 3;
    // #pragma parallel for
    for (unsigned int i = 0; i < 4; i++) {
        unsigned int point_fiber = (ndata_fiber * fiber_index) + (points[i] * 3);  // Mult by 3 dim
        unsigned int point_atlas = (ndata_fiber * fatlas_index) + (points[i] * 3);
        unsigned int point_inv_a = (ndata_fiber * fatlas_index) + (points[inv] * 3);
        float ed;
        if (!is_inverted) {
            ed = euclidean_distance(subject_data[point_fiber], subject_data[point_fiber + 1], subject_data[point_fiber + 2], atlas_data[point_atlas],
                                    atlas_data[point_atlas + 1], atlas_data[point_atlas + 2]);
        } else {
            ed = euclidean_distance(subject_data[point_fiber], subject_data[point_fiber + 1], subject_data[point_fiber + 2], atlas_data[point_inv_a],
                                    atlas_data[point_inv_a + 1], atlas_data[point_inv_a + 2]);
        }

        if (ed > (threshold * threshold)) return 1;
        inv--;
    }
    return 0;
}

__device__ float discarded_21points(float *subject_data, float *atlas_data, unsigned short int ndata_fiber, float threshold, int is_inverted,
                                    unsigned int fiber_index, unsigned int fatlas_index) {
    unsigned short int inv = 20;
    float ed;
    float max_ed = 0;
    for (unsigned short int i = 0; i < 21; i++) {
        unsigned int fiber_point = (ndata_fiber * fiber_index) + (i * 3);
        unsigned int atlas_point = (ndata_fiber * fatlas_index) + (i * 3);
        unsigned int point_inv = (ndata_fiber * fatlas_index) + (inv * 3);
        if (!is_inverted) {
            ed = euclidean_distance_norm(subject_data[fiber_point], subject_data[fiber_point + 1], subject_data[fiber_point + 2], atlas_data[atlas_point],
                                         atlas_data[atlas_point + 1], atlas_data[atlas_point + 2]);
        } else {
            ed = euclidean_distance_norm(subject_data[fiber_point], subject_data[fiber_point + 1], subject_data[fiber_point + 2], atlas_data[point_inv],
                                         atlas_data[point_inv + 1], atlas_data[point_inv + 2]);
        }

        if (ed > threshold) return -1;
        if (ed >= max_ed) max_ed = ed;
        inv--;
    }
    // After pass the comprobation of euclidean distance, will be tested with
    // the lenght factor
    unsigned int fiber_pos = (ndata_fiber * fiber_index);
    unsigned int atlas_pos = (ndata_fiber * fatlas_index);
    float length_fiber1 = euclidean_distance_norm(subject_data[fiber_pos], subject_data[fiber_pos + 1], subject_data[fiber_pos + 2],
                                                  subject_data[fiber_pos + 3], subject_data[fiber_pos + 4], subject_data[fiber_pos + 5]);
    float length_fiber2 = euclidean_distance_norm(atlas_data[atlas_pos], atlas_data[atlas_pos + 1], atlas_data[atlas_pos + 2], atlas_data[atlas_pos + 3],
                                                  atlas_data[atlas_pos + 4], atlas_data[atlas_pos + 5]);
    float fact = length_fiber2 < length_fiber1 ? ((length_fiber1 - length_fiber2) / length_fiber1) : ((length_fiber2 - length_fiber1) / length_fiber2);
    fact = (((fact + 1.0f) * (fact + 1.0f)) - 1.0f);
    fact = fact < 0.0f ? 0.0f : fact;

    if ((max_ed + fact) >= threshold)
        return -1;
    else
        return max_ed;
}

__global__ void parallel_segmentation_kernel(float *atlas_data, float *subject_data, unsigned int *bundle_of_fiber, float *thresholds, int *assignment,
                                             unsigned int atlas_size, unsigned int subject_size, unsigned short int ndata_fiber) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 0 && i < subject_size) {
        float ed_i = 500;
        int assignment_i = 65534;
        for (unsigned int j = 0; j < atlas_size; j++) {
            int is_inverted, is_discarded;
            float ed = -1;
            // unsigned int bundle = j;
            unsigned short b = bundle_of_fiber[j];
            // First test: discard_centers++; discard centroid
            is_discarded = discard_center(subject_data, atlas_data, ndata_fiber, thresholds[b], i, j);
            if (is_discarded == 1) continue;
            // Second test: discard by the extremes
            is_discarded = discard_extremes(subject_data, atlas_data, ndata_fiber, thresholds[b], &is_inverted, i, j);
            if (is_discarded == 1) continue;
            // Third test: discard by four points
            is_discarded = discard_four_points(subject_data, atlas_data, ndata_fiber, thresholds[b], is_inverted, i, j);
            if (is_discarded == 1) continue;
            ed = discarded_21points(subject_data, atlas_data, ndata_fiber, thresholds[b], is_inverted, i, j);
            if (ed != -1) {
                if (ed < ed_i) {
                    ed_i = ed;
                    assignment_i = b;
                }
            }
        }
        if (assignment_i != 65534) {
            if (assignment[i] == 65534) assignment[i] = assignment_i;
        }
    }
}
"""


def parallel_segmentation(
    subject_data, atlas_data, ndata_fiber, thresholds, bundle_of_fiber
):
    nfibers_subject = subject_data.shape[0] // ndata_fiber
    nfibers_atlas = atlas_data.shape[0] // ndata_fiber

    logger.info(f"Number of fibers: Subject {nfibers_subject}, Atlas {nfibers_atlas}")

    logger.debug("Creating np arrays")
    atlas_data = np.array(atlas_data, dtype=np.float32)
    subject_data = np.array(subject_data, dtype=np.float32)
    thresholds = np.array(thresholds, dtype=np.float32)
    bundle_of_fiber = np.array(bundle_of_fiber, dtype=np.uint32)
    assignment = np.full(nfibers_subject, 65534, dtype=np.int32)

    logger.debug("Creating SourceModule")
    mod = SourceModule(kernel_code)

    func = mod.get_function("parallel_segmentation_kernel")
    logger.debug("Calling parallel_segmentation_kernel")

    start_gpu = time.time()
    logger.debug("Allocating memory")
    logger.debug(
        f"MB: Atlas {atlas_data.nbytes / 1048576:.2f}, Subject {subject_data.nbytes / 1048576:.2f}, Bundle {bundle_of_fiber.nbytes / 1048576:.2f}, Threshold {thresholds.nbytes / 1048576:.2f}, Assignment {assignment.nbytes / 1048576:.2f}"
    )
    free_before, _ = cuda.mem_get_info()
    atlas_gpu = cuda.mem_alloc(atlas_data.nbytes)
    logger.debug("After mem alloc atlas_data")
    subject_gpu = cuda.mem_alloc(subject_data.nbytes)
    logger.debug("After mem alloc subject_data")
    bundle_gpu = cuda.mem_alloc(bundle_of_fiber.nbytes)
    logger.debug("After mem alloc bundle_of_fiber")
    thresh_gpu = cuda.mem_alloc(thresholds.nbytes)
    logger.debug("After mem alloc thresholds")
    assignment_gpu = cuda.mem_alloc(assignment.nbytes)  # resultados van aca
    logger.debug("After cuda mem alloc")

    cuda.memcpy_htod(atlas_gpu, atlas_data)
    cuda.memcpy_htod(subject_gpu, subject_data)
    cuda.memcpy_htod(bundle_gpu, bundle_of_fiber)
    cuda.memcpy_htod(thresh_gpu, thresholds)
    cuda.memcpy_htod(assignment_gpu, assignment)

    end_gpu = time.time()

    logger.info(f"Tiempo transferencia CPU → GPU: {end_gpu - start_gpu:.4f} segundos")

    start_kernel = time.time()
    block_size = 1024  # Maybe it's the best number, 1024 for better gpus
    grid_size = (nfibers_subject + block_size - 1) // block_size

    func(
        atlas_gpu,
        subject_gpu,
        bundle_gpu,
        thresh_gpu,
        assignment_gpu,
        np.uint32(nfibers_atlas),
        np.uint32(nfibers_subject),
        np.uint16(ndata_fiber),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )

    cuda.Context.synchronize()  # Espera a que termine el kernel
    free_after, _ = cuda.mem_get_info()
    used = free_before - free_after
    logger.info(f"Memoria usada: {used / (1024**2):.2f} MB")
    end_kernel = time.time()
    logger.info(
        f"Tiempo ejecución kernel (segmentación): {end_kernel - start_kernel:.4f} segundos"
    )

    start_cpu = time.time()
    cuda.memcpy_dtoh(assignment, assignment_gpu)
    end_cpu = time.time()
    logger.info(f"Tiempo transferencia GPU → CPU: {end_cpu - start_cpu:.4f} segundos")

    return assignment


# Lectura de archivos


def read_bundles(path, ndata_fiber):
    data = []
    with open(path, "rb") as f:
        content = f.read()
        fiber_size = 4 + ndata_fiber * 4
        nfibers = len(content) // fiber_size
        for i in range(nfibers):
            start = i * fiber_size
            raw = content[start + 4 : start + fiber_size]
            floats = struct.unpack("f" * ndata_fiber, raw)
            data.extend(floats)

    return np.array(data, dtype=np.float32)


def read_atlas_info(path):
    names, thresholds, fibers_per_bundle = [], [], []
    nfibers_atlas = 0
    with open(path, "r") as f:
        for line in f:
            name, threshold, n = line.strip().split()
            names.append(name)
            thresholds.append(float(threshold))
            fibers_per_bundle.append(int(n))
            nfibers_atlas += int(n)
    logger.info(
        f"Size of: names {len(names)}, thresholds {len(thresholds)}, fibers_per_bundle {len(fibers_per_bundle)}"
    )
    logger.info(f"nfibers_atlas {nfibers_atlas}")
    return names, thresholds, fibers_per_bundle, nfibers_atlas


def atlas_bundle(fibers_per_bundle):
    correspondence = []
    for idx, count in enumerate(fibers_per_bundle):
        correspondence.extend([idx] * count)
    logger.info(f"Size of correspondence: {len(correspondence)}")
    return correspondence


def read_atlas_bundles(atlas_data_dir, bundle_names, ndata_fiber, verbose=True):
    # Retorna:atlas_data: np.array con todas las fibras del atlas
    atlas_data = []
    for name in bundle_names:
        bundle_file = os.path.join(atlas_data_dir, name + ".bundlesdata")
        bundle_data = read_bundles(bundle_file, ndata_fiber)
        atlas_data.extend(bundle_data)

    atlas_data = np.array(atlas_data, dtype=np.float32)

    if verbose:
        logger.info(f"Total de fascículos cargados: {len(bundle_names)}")
        if thresholds is not None:
            logger.info(f"Umbral del primer fascículo: {thresholds[0]}")

    return atlas_data


def contar_fibras_segmentadas(assignment):
    return np.count_nonzero(assignment != 65534)


def write_bundles(
    output_dir, subject_name, assignment, bundle_names, ndata_fiber, subject_data
):
    os.makedirs(output_dir, exist_ok=True)
    for i, name in enumerate(bundle_names):
        indices = np.where(assignment == i)[0]
        fibras = np.empty((len(indices) * ndata_fiber,), dtype=np.float32)
        for j, idx in enumerate(indices):
            start = idx * ndata_fiber
            end = start + ndata_fiber
            fibras[j * ndata_fiber : (j + 1) * ndata_fiber] = subject_data[start:end]
        filepath = os.path.join(output_dir, f"{subject_name}_{name}.bundlesdata")
        fibras.tofile(filepath)


def write_indices(indices_dir, bundle_names, assignment):
    os.makedirs(indices_dir, exist_ok=True)
    for i, name in enumerate(bundle_names):
        indices = np.where(assignment == i)[0]
        np.savetxt(os.path.join(indices_dir, f"{name}.txt"), indices, fmt="%d")


def get_bundlesdata_files(path):
    bundlesdata_files = []

    for current_path, subfolders, files in os.walk(path):
        for file in files:
            if file.endswith(".bundlesdata"):
                bundlesdata_files.append(os.path.join(current_path, file))

    return bundlesdata_files


if __name__ == "__main__":
    n_points = 21
    ndata_fiber = n_points * 3
    n_runs_per_file = 3

    bundles_files = get_bundlesdata_files("files/sujetos")

    runtime_per_file = {}
    for input_path in bundles_files:
        for run in range(1, n_runs_per_file + 1):
            start_total = time.time()
            subject_path = input_path
            atlas_info_path = "files/AtlasRo_info.txt"
            atlas_data_dir = "files/AtlasRo"
            subject_name = input_path.split("/")[-2]
            output_dir = f"outputs/{run}_{subject_name}_segmentacion_resultados"
            indices_output_dir = f"outputs/{run}_{subject_name}_segmentacion_indices"

            logger.info(f"Subject: {subject_name}, Run: {run}")

            if subject_name not in runtime_per_file:
                runtime_per_file[subject_name] = []

            bundle_names, thresholds, fibers_per_bundle, nfibers_atlas = (
                read_atlas_info(atlas_info_path)
            )
            bundle_of_fiber = atlas_bundle(fibers_per_bundle)
            subject_data = read_bundles(subject_path, ndata_fiber)
            atlas_data = read_atlas_bundles(atlas_data_dir, bundle_names, ndata_fiber)

            logger.info(f"Total de fascículos: {len(bundle_names)}")
            logger.info(f"Umbral bundle[0]: {thresholds[0]}")

            assignment = parallel_segmentation(
                subject_data, atlas_data, ndata_fiber, thresholds, bundle_of_fiber
            )

            count = contar_fibras_segmentadas(assignment)
            logger.info(f"Fibras asignadas: {count} / {len(assignment)}")

            write_bundles(
                output_dir,
                subject_name,
                assignment,
                bundle_names,
                ndata_fiber,
                subject_data,
            )
            write_indices(indices_output_dir, bundle_names, assignment)
            end_total = time.time()
            logger.info(
                f"Tiempo total de segmentación (GPU): {end_total - start_total:.4f} segundos"
            )
            runtime_per_file[subject_name].append(end_total - start_total)
    logger.info(f"Runtime per file: {runtime_per_file}")