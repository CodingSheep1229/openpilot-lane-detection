#include <Eigen/Dense>
#include <string.h>
#include <fstream>
#include <iostream>
using namespace std;

#define MODEL_PATH_DISTANCE 100
#define POLYFIT_DEGREE 4

Eigen::Matrix<float, MODEL_PATH_DISTANCE, POLYFIT_DEGREE> vander;

void poly_fit(float *in_pts, float *in_stds, float *out) {
  // References to inputs
  Eigen::Map<Eigen::Matrix<float, MODEL_PATH_DISTANCE, 1> > pts(in_pts, MODEL_PATH_DISTANCE);
  Eigen::Map<Eigen::Matrix<float, MODEL_PATH_DISTANCE, 1> > std(in_stds, MODEL_PATH_DISTANCE);
  Eigen::Map<Eigen::Matrix<float, POLYFIT_DEGREE, 1> > p(out, POLYFIT_DEGREE);

  // Build Least Squares equations
  Eigen::Matrix<float, MODEL_PATH_DISTANCE, POLYFIT_DEGREE> lhs = vander.array().colwise() / std.array();
  Eigen::Matrix<float, MODEL_PATH_DISTANCE, 1> rhs = pts.array() / std.array();

  // Improve numerical stability
  Eigen::Matrix<float, POLYFIT_DEGREE, 1> scale = 1. / (lhs.array()*lhs.array()).sqrt().colwise().sum();
  lhs = lhs * scale.asDiagonal();

  // Solve inplace
  Eigen::ColPivHouseholderQR<Eigen::Ref<Eigen::MatrixXf> > qr(lhs);
  p = qr.solve(rhs);

  // Apply scale to output
  p = p.transpose() * scale.asDiagonal();
}

int main(int argc, char** argv)
{
	for(int i = 0; i < MODEL_PATH_DISTANCE; i++) {
    for(int j = 0; j < POLYFIT_DEGREE; j++) {
      vander(i, j) = pow(i, POLYFIT_DEGREE-j-1);
    }
  }
  float* points = new float[MODEL_PATH_DISTANCE]();
  float* stds = new float[MODEL_PATH_DISTANCE]();
  float* poly = new float[POLYFIT_DEGREE]();
  string filename = argv[1];
  string name = filename + "_pts";
  fstream file(name.c_str());
  float tp;
  int idx=0;
  while(file >> tp)
  {
    // cout << tp << " ";
    points[idx] = tp;
    idx++;
  }
  cout << endl;
  file.close();
  name = filename+"_stds";
  fstream file2(name.c_str());
  idx=0;
  while(file2 >> tp)
  {
    // cout << tp << " ";
    stds[idx] = tp;
    idx++;
  }

  poly_fit(points, stds, poly);
  cout << poly[0];
  for (int i=1;i<POLYFIT_DEGREE;i++)
  {
    cout << " " << poly[i]; 
  }
  cout << endl;
	return 0;
}