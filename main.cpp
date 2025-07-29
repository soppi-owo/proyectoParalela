/*# Copyright (C) 2019  Andrea Vázquez Varela

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

 main.cpp
Authors:
    Narciso López López
    Andrea Vázquez Varela
Last modification: 24-10-2018 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>

#include "dirent.h"

using namespace std;

float sqrt7(float x) {
  unsigned int i = *(unsigned int *)&x;
  i += 127 << 23; // adjust bias
  i >>= 1;        // approximation of square root
  return *(float *)&i;
}

float euclidean_distance(float x1, float y1, float z1, float x2, float y2,
                         float z2) {
  return ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) +
          (z1 - z2) * (z1 - z2));
}

/*Calculate the euclidean distance between two 3d points normalized*/
float euclidean_distance_norm(float x1, float y1, float z1, float x2, float y2,
                              float z2) {
  return sqrt7((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) +
               (z1 - z2) * (z1 - z2));
}

/*Return true when the fiber is discarded measuring the distance in central
 * points*/
bool discard_center(vector<float> &subject_data, vector<float> &atlas_data,
                    unsigned short int ndata_fiber, float threshold,
                    unsigned int fiber_index, unsigned int fatlas_index) {
  unsigned int fpoint = (fiber_index * ndata_fiber) +
                        31; // Point of fiber, 31 is the middle of the fiber
  unsigned int apoint = (fatlas_index * ndata_fiber) +
                        31; // Atlas point, 31 is the middle of the fiber

  float ed = euclidean_distance(
      subject_data[fpoint - 1], subject_data[fpoint], subject_data[fpoint + 1],
      atlas_data[apoint - 1], atlas_data[apoint], atlas_data[apoint + 1]);
  if (ed > (threshold * threshold))
    return true;
  else
    return false;
}

bool discard_extremes(vector<float> &subject_data, vector<float> &atlas_data,
                      unsigned short int ndata_fiber, float threshold,
                      bool &is_inverted, unsigned int fiber_index,
                      unsigned int fatlas_index) {
  unsigned int fpoint1 = fiber_index * ndata_fiber;  // Point 0 of fiber
  unsigned int apoint1 = fatlas_index * ndata_fiber; // Atlas point 0
  unsigned int fpoint21 = fpoint1 + 62;              // Last point on the fiber
  unsigned int apoint21 = apoint1 + 62;              // Last point on the fiber
  float first_points_ed_direct =
      euclidean_distance(subject_data[fpoint1], subject_data[fpoint1 + 1],
                         subject_data[fpoint1 + 2], atlas_data[apoint1],
                         atlas_data[apoint1 + 1], atlas_data[apoint1 + 2]);
  float first_point_ed_flip =
      euclidean_distance(subject_data[fpoint1], subject_data[fpoint1 + 1],
                         subject_data[fpoint1 + 2], atlas_data[apoint21 - 2],
                         atlas_data[apoint21 - 1], atlas_data[apoint21]);
  float first_points_ed = min(first_points_ed_direct, first_point_ed_flip);

  if (first_points_ed > (threshold * threshold))
    return true;
  else {
    float last_points_ed;
    if (first_points_ed_direct < first_point_ed_flip) {
      is_inverted = false;
      last_points_ed = euclidean_distance(
          subject_data[fpoint21 - 2], subject_data[fpoint21 - 1],
          subject_data[fpoint21], atlas_data[apoint21 - 2],
          atlas_data[apoint21 - 1], atlas_data[apoint21]);
    } else {
      is_inverted = true;
      last_points_ed = euclidean_distance(
          subject_data[fpoint21 - 2], subject_data[fpoint21 - 1],
          subject_data[fpoint21], atlas_data[apoint1], atlas_data[apoint1 + 1],
          atlas_data[apoint1 + 2]);
    }
    if (last_points_ed > (threshold * threshold))
      return true;
    else
      return false;
  }
}

bool discard_four_points(vector<float> &subject_data, vector<float> &atlas_data,
                         unsigned short int ndata_fiber, float threshold,
                         bool is_inverted, unsigned int fiber_index,
                         unsigned int fatlas_index) {
  vector<unsigned short int> points = {3, 7, 13, 17};
  unsigned short int inv = points.size() - 1;
  for (unsigned int i = 0; i < points.size(); i++) {
    unsigned int point_fiber =
        (ndata_fiber * fiber_index) + (points[i] * 3); // Mult by 3 dim
    unsigned int point_atlas = (ndata_fiber * fatlas_index) + (points[i] * 3);
    unsigned int point_inv_a = (ndata_fiber * fatlas_index) + (points[inv] * 3);
    float ed;
    if (!is_inverted) {
      ed = euclidean_distance(
          subject_data[point_fiber], subject_data[point_fiber + 1],
          subject_data[point_fiber + 2], atlas_data[point_atlas],
          atlas_data[point_atlas + 1], atlas_data[point_atlas + 2]);
    } else {
      ed = euclidean_distance(
          subject_data[point_fiber], subject_data[point_fiber + 1],
          subject_data[point_fiber + 2], atlas_data[point_inv_a],
          atlas_data[point_inv_a + 1], atlas_data[point_inv_a + 2]);
    }

    if (ed > (threshold * threshold))
      return true;
    inv--;
  }
  return false;
}

float discarded_21points(vector<float> &subject_data, vector<float> &atlas_data,
                         unsigned short int ndata_fiber, float threshold,
                         bool is_inverted, unsigned int fiber_index,
                         unsigned int fatlas_index) {
  unsigned short int inv = 20;
  float ed;
  float max_ed = 0;
  for (unsigned short int i = 0; i < 21; i++) {
    unsigned int fiber_point = (ndata_fiber * fiber_index) + (i * 3);
    unsigned int atlas_point = (ndata_fiber * fatlas_index) + (i * 3);
    unsigned int point_inv = (ndata_fiber * fatlas_index) + (inv * 3);
    if (!is_inverted) {
      ed = euclidean_distance_norm(
          subject_data[fiber_point], subject_data[fiber_point + 1],
          subject_data[fiber_point + 2], atlas_data[atlas_point],
          atlas_data[atlas_point + 1], atlas_data[atlas_point + 2]);
    } else {
      ed = euclidean_distance_norm(
          subject_data[fiber_point], subject_data[fiber_point + 1],
          subject_data[fiber_point + 2], atlas_data[point_inv],
          atlas_data[point_inv + 1], atlas_data[point_inv + 2]);
    }

    if (ed > threshold)
      return -1;
    if (ed >= max_ed)
      max_ed = ed;
    inv--;
  }
  // After pass the comprobation of euclidean distance, will be tested with the
  // lenght factor
  unsigned int fiber_pos = (ndata_fiber * fiber_index);
  unsigned int atlas_pos = (ndata_fiber * fatlas_index);
  float length_fiber1 = euclidean_distance_norm(
      subject_data[fiber_pos], subject_data[fiber_pos + 1],
      subject_data[fiber_pos + 2], subject_data[fiber_pos + 3],
      subject_data[fiber_pos + 4], subject_data[fiber_pos + 5]);
  float length_fiber2 = euclidean_distance_norm(
      atlas_data[atlas_pos], atlas_data[atlas_pos + 1],
      atlas_data[atlas_pos + 2], atlas_data[atlas_pos + 3],
      atlas_data[atlas_pos + 4], atlas_data[atlas_pos + 5]);
  // condición ? valor_si_verdadero : valor_si_falso;
  float fact = length_fiber2 < length_fiber1
                   ? ((length_fiber1 - length_fiber2) / length_fiber1)
                   : ((length_fiber2 - length_fiber1) / length_fiber2);
  fact = (((fact + 1.0f) * (fact + 1.0f)) - 1.0f);
  fact = fact < 0.0f ? 0.0f : fact; // si fact menor a 0, entonces fact es 0

  if ((max_ed + fact) >= threshold)
    return -1;
  else // cumple condicion
    return max_ed;
}

void write_indices(const std::string &path, vector<string> &names,
                   const std::vector<std::vector<float>> &ind) {
  DIR *dir;
  if ((dir = opendir(path.c_str())) ==
      NULL) { // Checks if a directory path exists

    int dir_err = filesystem::create_directories(path);
    if (dir_err == -1) {
      perror("Error creating directory!");
      exit(EXIT_FAILURE);
    }
  }
  closedir(dir);

  for (uint32_t i = 0; i < ind.size(); i++) {
    if (ind[i].size() == 0) {
      continue;
    }

    std::ofstream file(path + "/" + names[i] + ".txt", std::ios::out);

    if (file.is_open()) {
      for (uint32_t j = 0; j < ind[i].size(); j++)
        file << fixed << ind[i][j] << std::endl;
      // file.write(&ind[i][j], sizeof( uint8_t ));
    }

    file.close();
  }
}

/*Read .bundles files and return (by reference) a vector with the datas*/
void write_bundles(string subject_name, string output_path,
                   vector<vector<float>> &assignment, vector<string> &names,
                   int ndata_fiber, vector<float> &subject_data) {
  int npoints = ndata_fiber / 3;
  ofstream bundlesfile;
  struct stat sb;
  if (stat(output_path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
    string command = "rm -r " + output_path;
    int del = system(command.c_str());
  }
  filesystem::create_directories(output_path);
  for (unsigned int i = 0; i < assignment.size(); i++) {
    if (assignment[i].size() != 0) {
      string bundlesdata_path =
          output_path + "/" + subject_name + "_to_" + names[i] + ".bundlesdata";
      FILE *fp = fopen(bundlesdata_path.c_str(), "wb"); // Opening and writing .bundlesdata file.
      if (fp == NULL) {
        fputs("File error opening .bundlesdata file\n", stderr);
        exit(1);
      }
      for (unsigned int j = 0; j < assignment[i].size(); j++) {
        int fiber_index = assignment[i][j];
        fwrite(&npoints, sizeof(uint32_t), 1, fp);

        // cout << &subject_data[fiber_index*ndata_fiber] << endl;

        fwrite(&subject_data[fiber_index * ndata_fiber], sizeof(float),
               ndata_fiber, fp);
      }
      fclose(fp);
      bundlesfile.open(output_path + "/" + subject_name + "_to_" + names[i] +
                           ".bundles",
                       ios::out);
      bundlesfile << "attributes = {" << endl
                  << "    \'binary\' : 1," << endl
                  << "    \'bundles\' : [ '" << (names[i]) << "', 0 ]," << endl
                  << "    \'byte_order\' : \'DCBA\'," << endl
                  << "    \'curves_count\' : " << assignment[i].size() << ","
                  << endl
                  << "    \'data_file_name\' : \'*.bundlesdata\'," << endl
                  << "    \'format\' : \'bundles_1.0\'," << endl
                  << "    \'space_dimension\' : 3" << endl
                  << "  }" << endl;
      bundlesfile.close();
    }
  }
}

/*Read .bundles files and return (by reference) a vector with the datas*/
vector<float> read_bundles(string path, unsigned short int ndata_fiber) {
  vector<float> data;
  char path2[path.length() + 1];
  strncpy(path2, path.c_str(), sizeof(path2));
  path2[sizeof(path2) - 1] = 0;
  FILE *fp = fopen(path2, "rb");
  // Open subject file.
  if (fp == NULL) {
    fputs("File error opening file\n", stderr);
    exit(1);
  }
  fseek(fp, 0, SEEK_END);
  long lSize = ftell(fp); // Get file size.
  unsigned int sfiber =
      sizeof(uint32_t) +
      ndata_fiber *
          sizeof(float); // Size of a fiber (bytes).  // Add 1 element
                         // (uint32_t) because in .bundles/.bundlesdata format
                         // the first element of each fiber/centroid corresponds
                         // to the amount of points in the fiber/centroid. In
                         // this case that number should be always the same.
  float buffer[sfiber];
  unsigned int nFibers = lSize / (float)sfiber; // Number of fibers
  rewind(fp);
  for (unsigned int j = 0; j < (nFibers); ++j) // Copy fibers.
  {
    int r = fread(buffer, sizeof(float), (ndata_fiber + 1),
                  fp); // Skip the first element of each fiber/centroid (number
                       // of points).;
    if (r == -1)
      cout << "error reading buffer data";
    for (int s = 1; s < ndata_fiber + 1; ++s) {
      data.push_back(buffer[s]);
    }
  }

  fclose(fp);
  return data;
}

/*Get vector of bundles of the atlas*/
vector<float> get_atlas_bundles(string path, vector<string> names,
                                unsigned short int ndata_fiber) {
  vector<float> atlas_bundles;
  for (unsigned int i = 0; i < names.size(); i++) {
    string file_path = path + "/" + names[i] + ".bundlesdata";

    vector<float> bundle = read_bundles(file_path, ndata_fiber);
    atlas_bundles.insert(atlas_bundles.end(), bundle.begin(), bundle.end());
  }
  return atlas_bundles;
}

/*Read atlas information file*/
void read_atlas_info(string path, vector<string> &names, vector<float> &thres,
                     unsigned int &nfibers_atlas,
                     vector<unsigned int> &fibers_per_bundle) {
  ifstream infile(path, ios::in);
  if (!infile)
    cerr << "Cant open " << endl;

  string name;
  float t;
  unsigned int n;
  while (infile >> name >> t >> n) {
    names.push_back(name);
    thres.push_back(t);
    nfibers_atlas += n;
    fibers_per_bundle.push_back(n);
    // cout<< name << " "<< to_string(t)<<" "<< to_string(n) << endl;
  }

  /*for(uint32_t jaja = 0; jaja < names.size(); jaja++){
      std::cout << names[jaja] << std::endl;
  }*/
}
// funcion que establece a que fasciculo corresponde cada fibra del atlas
vector<unsigned int> atlas_bundle(vector<unsigned int> &fibers_per_bundle,
                                  unsigned int nfibers) {
  vector<unsigned int> correspondence(nfibers);
  unsigned int fiber_index = 0;
  // for i que recorre los fasciculos
  for (unsigned int i = 0; i < fibers_per_bundle.size(); i++) {
    // for j que recorre las fibras de cada fasciculo
    for (unsigned int j = 0; j < fibers_per_bundle[i]; j++) {
      correspondence[fiber_index] = i;
      fiber_index++;
    }
  }
  return correspondence;
}

bool sort_by_length(float i, float j) { return (i < j); }

vector<unsigned short>
parallel_segmentation(vector<float> &atlas_data, vector<float> &subject_data,
                      unsigned short int ndata_fiber, vector<float> thresholds,
                      vector<unsigned int> &bundle_of_fiber) {
  unsigned int nfibers_subject = subject_data.size() / ndata_fiber;
  unsigned int nfibers_atlas = atlas_data.size() / ndata_fiber;
  int contador_fibras = 1;
  vector<unsigned short> assignment(
      nfibers_subject, 65534); // vector donde cada valor indica el fasciulo al
                               // que pertenece la fibra i
  unsigned int nunProc = omp_get_num_procs();
  // se deja como numero de hilos el total de procesadores disponibles
  omp_set_num_threads(nunProc);
  cout << "procesadores: " << nunProc << endl << flush;
  cout << "entro a parallel" << endl << flush;
#pragma omp parallel
  {
#pragma omp for schedule(auto) nowait
    // se compara cada fibra i del sujeto con cada fibra j del atlas
    for (unsigned long i = 0; i < nfibers_subject; i++) {
      float ed_i = 500; //??
      unsigned short assignment_i = 65534;
      for (unsigned int j = 0; j < nfibers_atlas; j++) {
        bool is_inverted, is_discarded;
        float ed = -1;
        unsigned short b =
            bundle_of_fiber[j]; // el arreglo de tamaño de todas las fibras me
                                // dice por posición a qué fascículo pertenece
                                // cada fibra, entonces me da el fasciculo al
                                // que pertenece la fibra segmentada
        is_discarded = discard_center(subject_data, atlas_data, ndata_fiber,
                                      thresholds[b], i, j);
        if (is_discarded)
          continue;
        // segundo filtro, extremos
        is_discarded = discard_extremes(subject_data, atlas_data, ndata_fiber,
                                        thresholds[b], is_inverted, i, j);
        if (is_discarded)
          continue;
        // tercer filtro, 4 puntos
        is_discarded =
            discard_four_points(subject_data, atlas_data, ndata_fiber,
                                thresholds[b], is_inverted, i, j);
        if (is_discarded)
          continue;
        ed = discarded_21points(subject_data, atlas_data, ndata_fiber,
                                thresholds[b], is_inverted, i, j);

        if (ed != -1) {
          if (ed < ed_i) {
            ed_i = ed;
            assignment_i = b;
          }
        }
      }
      if (assignment_i != 65534) {
        assignment[i] = assignment_i;
      }
    }
  }
  return assignment;
}

vector<string> getBundlesdataFiles(string path) {
  vector<string> bundlesdataFiles;

  for (const auto &entry : filesystem::recursive_directory_iterator(path)) {
    if (entry.is_regular_file() && entry.path().extension() == ".bundlesdata") {
      bundlesdataFiles.push_back(entry.path().string());
    }
  }

  return bundlesdataFiles;
}

int main() {
  unsigned short int n_points = 21;
  unsigned short int ndata_fiber = n_points * 3;
  unsigned short int n_runs_per_file = 3;

  vector<string> bundles_files = getBundlesdataFiles("files/sujetos");

  map<string, vector<double>> runtime_per_file;

  for (int input_path = 0; input_path < bundles_files.size(); input_path++) {
    for (int run = 1; run <= n_runs_per_file; run++) {
      double time_start, time_start_paralell, parallelFastCPUTime, final_time;
      time_start = omp_get_wtime();
      string subject_path = bundles_files[input_path];
      string subject_name =
          filesystem::path(subject_path).parent_path().filename();
      string atlas_path = "files/AtlasRo";
      string atlas_inf = "files/AtlasRo_info.txt";
      string output_dir = "outputs/" + std::to_string(run) + "_" + subject_name +
                          "_ResultadosCpp";
      string indices_output_dir =
          "outputs/" + std::to_string(run) + "_" + subject_name + "_IndicesCpp";

      cout << "Ejecutando Run " << run << " para archivo " << subject_path
           << endl;
      cout << "\tSubject Name: " << subject_name << "\n"
           << "\tOutput Dir: " << output_dir << "\n"
           << "\tIndices Output Dir: " << indices_output_dir << "\n";

      // usamos 21 puntos por fibra
      // coordenadas

      cout << "accedi directorios" << endl << flush;

      // Atlas data
      // el umbral tambien siempre es 8
      vector<float> thresholds;
      vector<string> bundles_names;
      unsigned int nfibers_atlas = 0; // cuantas fibras tiene el atlas
      vector<unsigned int> fibers_per_bundle;
      vector<unsigned int> bundle_of_fiber; //??
      vector<float> atlas_data;
      vector<vector<float>> atlas_centroids; // creo que no lo estamos usando

      // Subject data
      vector<float> subject_data;

      // Read the atlas information file and get the number of bundles of the
      // atlas
      read_atlas_info(atlas_inf, bundles_names, thresholds, nfibers_atlas,
                      fibers_per_bundle);
      bundle_of_fiber = atlas_bundle(
          fibers_per_bundle, nfibers_atlas); // vector de correspondencia entre
                                             // fibras y fasciculos del atlas
      cout << "leo atlas" << endl << flush;

      // Read the atlas data of .bundledata files and create the vectors of
      // bundles
      atlas_data = get_atlas_bundles(atlas_path, bundles_names, ndata_fiber);
      cout << "lei atlas, empiezo sujeto" << endl << flush;

      // leer sujeto
      subject_data = read_bundles(subject_path, ndata_fiber);

      cout << "lei sujeto" << endl << flush;
      vector<unsigned short> assignment;
      time_start_paralell = omp_get_wtime();

      assignment = parallel_segmentation(atlas_data, subject_data, ndata_fiber,
                                         thresholds, bundle_of_fiber);

      // std::cout << static_cast<unsigned>(assignment) << std::endl;
      // vector<int> assignment =
      // parallel_segmentation(atlas_centroids,subject_data,ndata_fiber,thresholds);

      parallelFastCPUTime = omp_get_wtime() - time_start_paralell;

      vector<vector<float>> map_results(bundles_names.size());

      // Map assignment
      for (unsigned int j = 0; j < assignment.size(); j++) {
        if (assignment[j] != 65534) {
          if (assignment[j] >= map_results.size()) {
            cerr << "[ERROR] assignment[" << j << "] = " << assignment[j]
                 << " está fuera del rango. bundles_names.size() = "
                 << map_results.size() << endl;
            exit(1); // Detiene el programa para depurar si ocurre
          }
          map_results[assignment[j]].push_back(j);
        }
      }
      int count = 0;
      for (unsigned int i = 0; i < assignment.size(); i++) {
        if (assignment[i] != 65534) {
          count++;
        }
      }
      write_bundles(subject_name, output_dir, map_results, bundles_names,
                    ndata_fiber, subject_data);
      write_indices(
          indices_output_dir, bundles_names,
          map_results); // este antes estaba fuera del conteo de tiempo
      final_time = omp_get_wtime() - time_start;

      cout << "Total segmented fibers: " << to_string(count) << endl << endl;
      cout << "Execution time of fast algorithm (Parallel version): "
           << parallelFastCPUTime << endl;
      cout << "Execution time IO operations: "
           << final_time - parallelFastCPUTime << endl;
      cout << "Total time execution: " << final_time << endl;
      runtime_per_file[subject_name].push_back(parallelFastCPUTime);
    }
  }

  for (const auto &par : runtime_per_file) {
    cout << par.first << ": ";
    for (double tiempo : par.second) {
      cout << tiempo << ", ";
    }
    cout << endl;
  }

  return 0;
}