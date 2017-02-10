#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <fstream>
#include <string>
#include <iomanip>

//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "dirent.h"

#include <unistd.h>
#include <sys/times.h>
#include <sys/time.h>

#include "tinyxml.h"
#include "tinystr.h"

using namespace std;
using namespace cv;

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

typedef vector <Point>  VertVect;      // this is how trail poly is represented

struct DistortParams
{
  int x_left_new, x_right_new;                // encodes horizontal shift, scaling
  bool do_horizontal_flip;                    // do flip?
  float contrast_factor, brightness_factor;   // encodes value scaling, shift of each pixel channel

};

// this covers live and dead scallops, plus fish, sharks, skates, starfish, etc.

struct ScallopParams
{
  Point p_annotation;                   // annotator's click 
  Point p_upper_left, p_lower_right;    // bounding box
  int type;
  bool has_scale;                       // whether p_upper_left, p_lower_right have been filled in
};

struct TreeParams
{
  Point v_bottom;    // bottom center of trunk
  float dx, dy;      // unit vector along trunk major axis
  float width;       // length of minor axis
};

typedef vector <TreeParams> TreeVect;

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

bool line_segment_intersection(Point, Point, Point, Point, Point &);
bool ray_line_segment_intersection(Point, Point, Point, Point, double, Point &);
bool ray_image_boundaries_intersection(Point, Point, double, Point &);
float point_line_distance(Point, Point, Point);

Mat apply_precomputed_distortion(DistortParams &, Mat &, const vector <int> &, vector <int> &, const vector <int> &, vector <int> &);

void print_distortion(FILE *, DistortParams &);

void dyn_save_vertfile(int);
void dyn_onKeyPress(char, bool=false);
int snap_y(int);
void set_current_index(int);

void onKeyPress(char, bool=false);
Mat smart_imread(string &);
string fullpathname_from_logdir_and_seqnum(string &, int);

void compute_trajectory();
int seqnum_from_signature(string &);
string logdir_from_signature(string &);

void add_images(string, vector <string> &);
void add_all_images_from_file(string);
void trail_draw_overlay();
void trail_onMouse(int, int, int, int, void *);

void tree_draw_overlay();
void tree_onMouse(int, int, int, int, void *);

void scallop_draw_overlay();
void scallop_onMouse(int, int, int, int, void *);

bool getScallopSignature(string, string &);

void saveVertMap(); int loadVertMap();
void saveBadMap(); int loadBadMap();

void saveVertVect(FILE *, VertVect &);
void saveVertVectNoParens(FILE *, VertVect &);

void saveScallopMap(); int loadScallopMap();
int loadScallopMap(bool);
void write_traintest_scallop(string = "/tmp/scallop_data", float = 0.8);
void compute_stats_traintest_scallop();
bool filter_for_traintest_scallop(ScallopParams &);

int most_isolated_nonvert_image_idx();
int most_isolated_nonvert_image_idx(int);

bool isBad(int);
bool isVert(int);

void write_traintest_vanilla_grayscale(string = "/tmp/tf_data", float = 0.8);
void write_traintest_chocolate_color(string = "/tmp/tf_data", float = 0.8);
void write_traintest_strawberry_color(string = "/tmp/tf_data", float = 0.8);
void write_traintest_nontrail_color(string = "/tmp/tf_data", float = 0.8);
void write_traintest_peach_color(string = "/tmp/tf_data", float = 0.8);

string fullpathname_to_signature(string &);

vector <Point> load_xml_polygon(string, float = 1.0);

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#define RAY_DELTA_THRESH 20    // how far we must move from initial click before
                               // using 2nd point to define ray
#define Y_TOLERANCE      3

#define IMAGE_ROW_FAR    100
#define IMAGE_ROW_NEAR   175
#define ZERO_INDEX       0
#define BIG_INDEX_STEP   10
#define REQUIRED_NUMBER_OF_VERTS_PER_IMAGE   4
#define NO_INDEX         -1
#define NO_DISTANCE      -1
#define COMMENT_CHAR     '#'
#define SIMILARITY_THRESHOLD  0.05
#define MAX_STRLEN       256
#define MONTH_DAY_OFFSET 12
#define MOST_ISOLATED_COVERAGE_FRACTION  0.01
#define NUM_SEQ_DIGITS   6

// make this 30
#define PEACH_SEQUENCE_LENGTH 30  // how many images BEFORE current one to save 
//#define PEACH_SEQUENCE_LENGTH 5  // how many images BEFORE current one to save 

//#define OUTPUT_STRAWBERRY_DATA
#define OUTPUT_PEACH_DATA

#define DYN_SEQ_LENGTH   4
#define DYN_SEQ_INTERVAL 10

#define FULL_DYN_SEQ_LENGTH  31

#define TRAIL_MODE    1
#define TREE_MODE     2
#define SCALLOP_MODE  3

#define SCALLOP_ALIVE_TYPE 1
#define SCALLOP_DEAD_TYPE  2
#define SHARK_TYPE         3    // also "Dogfish shark"
#define FISH_TYPE          4
#define SKATE_TYPE         5
#define STARFISH_TYPE      6
// also "Can", "Squid"
// also empty, "sc"

#define SCALLOP_UP    1.6   // from Hunter's web coords to image coords

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

//int object_input_mode = TREE_MODE;  // TRAIL_MODE;
int object_input_mode = SCALLOP_MODE;  // TRAIL_MODE;

bool do_full_dyn = false;

string dyn_filename_stem;  
string current_imsig;
int dyn_current_seq_idx;

vector <string> DistortDynShortname;
vector <string> DistortDynSig;
vector <DistortParams> DistortDynParams;

vector <FILE *> DynOutFP;

vector <string> DynShortname;
vector <string> DynSig;
vector < vector <string> > DynFullpathname;  // full-path image filenames for dyn: 0, 10, 20, 30
vector < vector < VertVect > > DynVert;      // actual trail vertices for each image 

string s_imagedirs;

vector <string> Fullpathname_vect;  // full-path image filenames

vector <string> scallop_csv_lines;
vector <string> scallop_csv_line_endings;

vector <string> scallop_Fullpathname_vect;  // full-path image filenames
set <string> scallop_Fullpathname_set;
vector <ScallopParams> scallop_params_vect;
map <int, string> scallop_idx_signature_map;
map <int, int> scallop_idx_line_idx_map;
map <int, int> scallop_line_idx_idx_map;

Point p_scallop_start, p_scallop_end;
Point p_mouse, p_mouse_dx, p_mouse_dy;

Mat current_im, draw_im;

vector <int> Random_idx;         // R[i] holds random index
vector <int> Nonrandom_idx;      // R[N[i]] = i
//vector < vector <Point> > Vert;  // actual trail vertices for each image 
vector < VertVect > Vert;  // actual trail vertices for each image 

//-------------------------------------------------------------------

vector <TreeVect>  Trees;   // actual tree info for each image

set <int> Tree_idx_set;          // which images have tree info
set <int> NoTree_idx_set;        // which images do NOT have tree info (Tree U NoTree = all images)

//-------------------------------------------------------------------

vector <int> ClosestVert_dist;   // for NoVert images, how many indices away is the *nearest* Vert image?
                                 // max of this is "most isolated"

map <string, int> Signature_idx_map;
vector <string> Idx_signature_vect;

set <int> Bad_idx_set;           // which images are bad
set <int> Vert_idx_set;          // which images have edge vertex info
set <int> NoVert_idx_set;        // which images do NOT have edge vertex info (Vert U NoVert = all images)

set <int> FilteredVert_idx_set;  // vert images that pass various tests
set <int> FilteredBad_idx_set;   // bad images that pass various tests

vector <string> External_bad_sig_vect;  // bad images not in current dataset
vector < pair <string, VertVect > > External_vert_vect;   // sigs + vertvects

set <string> dayofweek_set;

bool do_random = false;
bool do_verts = false;
bool do_overlay = true;
bool do_bad = false;
int bad_start, bad_end;          // indices of bad range

bool bad_current_index = false;
bool vert_current_index = false;

bool callbacks_set = false;
bool dragging = false;
bool erasing = false;
int dragging_x, dragging_y;

Point p_tree_bottom;   // bottom
Point p_tree_upper;    // point such that upper - bottom = ray direction of major axis of trunk
Point p_tree_inter;    // where trunk ray intersects image boundary
bool p_tree_inter_result = false;
bool p_tree_have_direction = false;
double p_tree_width_val;
double tree_dx_ortho, tree_dy_ortho;   // unit vector defining direction orthogonal to trunk direction

bool editing_tree = false;

Point p_topleft, p_topright, p_bottomleft, p_bottomright;

double fontScale = 0.35;

vector <int> trailEdgeRow;

int num_saved_verts = 0;
int current_index;
string current_imname;

int max_closest_vert_dist;
int next_nonvert_idx = NO_INDEX;

bool do_show_crop_rect = false;

#define CANONICAL_IMAGE_WIDTH        480
#define CANONICAL_IMAGE_HEIGHT       320

#define TOP_ONLY_TEST_TOP_Y          50
#define TOP_ONLY_TEST_WIDTH          250
#define TOP_ONLY_TEST_HEIGHT         100

//#define TOP_ONLY_TEST_OUTPUT_WIDTH   80
//#define TOP_ONLY_TEST_OUTPUT_HEIGHT  32
#define TOP_ONLY_TEST_OUTPUT_WIDTH   160
#define TOP_ONLY_TEST_OUTPUT_HEIGHT  64

// top is 100, bottom is 175
#define TOP_AND_BOTTOM_TEST_TOP_Y   59   // 50
#define TOP_AND_BOTTOM_TEST_WIDTH   250
//#define TOP_AND_BOTTOM_TEST_HEIGHT   175
#define TOP_AND_BOTTOM_TEST_HEIGHT   125
#define TOP_AND_BOTTOM_TEST_OUTPUT_WIDTH   160
//#define TOP_AND_BOTTOM_TEST_OUTPUT_HEIGHT  112
#define TOP_AND_BOTTOM_TEST_OUTPUT_HEIGHT  80

// chocolate
#ifdef OUTPUT_CHOCOLATE_DATA
int output_crop_top_y = TOP_ONLY_TEST_TOP_Y;  // 25;
int output_crop_width = TOP_ONLY_TEST_WIDTH; // 450;  // 375
int output_crop_height = TOP_ONLY_TEST_HEIGHT; // 225;   // 150
int output_width = TOP_ONLY_TEST_OUTPUT_WIDTH; // TEST_IMAGE_SIZE;  // 150;
int output_height = TOP_ONLY_TEST_OUTPUT_HEIGHT; // TEST_IMAGE_SIZE;  // 75;
#else   // strawberry OR peach
int output_crop_top_y = TOP_AND_BOTTOM_TEST_TOP_Y;  // 25;
int output_crop_width = TOP_AND_BOTTOM_TEST_WIDTH; // 450;  // 375
int output_crop_height = TOP_AND_BOTTOM_TEST_HEIGHT; // 225;   // 150
int output_width = TOP_AND_BOTTOM_TEST_OUTPUT_WIDTH; // TEST_IMAGE_SIZE;  // 150;
int output_height = TOP_AND_BOTTOM_TEST_OUTPUT_HEIGHT; // TEST_IMAGE_SIZE;  // 75;
#endif

#define MNIST_IMAGE_SIZE   28
#define TEST_IMAGE_SIZE      64
//#define MNIST_IMAGE_SIZE   100

#define SCALLOP_IMAGE_WIDTH        1280
#define SCALLOP_IMAGE_HEIGHT       960


int distort_max_horizontal_delta = 75;
float distort_horizontal_flip_prob = 0.5;
float distort_max_contrast_delta = 0.5;
float distort_max_brightness_delta = 50;
int distort_num_per_image = 10;    // 10
int nontrail_distort_num_per_image = 6;

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

//! generate random number uniformly distributed between [0.0, 1.0)

double uniform_UD_Random()
{
  // windows

  //  return (double) rand() / (double) RAND_MAX;

  // linux

  return drand48();
  //  UD_error("no uniform_UD_Random()");
  //  return 0;
}

//----------------------------------------------------------------------------

//! returns true with probability prob; false with probability 1 - prob

bool probability_UD_Random(double prob)
{
  return uniform_UD_Random() <= prob;
}

//----------------------------------------------------------------------------

double UD_round(double x)
{
  if (x - floor(x) >= 0.5)
    return ceil(x);
  else
    return floor(x);
}

//----------------------------------------------------------------------------

//! random uniform integer between lower and upper, inclusive

int ranged_uniform_int_UD_Random(int lower, int upper)
{
  double result;
  double range_size;

  range_size = upper - lower;
  result = range_size * uniform_UD_Random();
  result += lower;

  // round off

  result = UD_round(result);

  return (int) result;
}

//----------------------------------------------------------------------------

//! random number floating point number uniformly distributed between [lower, upper)

double ranged_uniform_UD_Random(double lower, double upper)
{
  double range_size;
  double result;

  range_size = upper - lower;
  result = range_size * uniform_UD_Random();
  result += lower;

  return result;
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

bool by_y (Point p, Point q) { return (p.y < q.y); }
bool by_x (Point p, Point q) { return (p.x < q.x); }

void sort_vertices(VertVect & V)
{
  //  for (int i = 0; i < V.size(); i++)
  //    printf("B %i: (%i, %i)\n", i, V[i].x, V[i].y);

  sort(V.begin(), V.end(), by_y);
  sort(V.begin(), V.begin()+1, by_x);
  sort(V.begin()+2, V.begin()+3, by_x);

  //  for (int i = 0; i < V.size(); i++)
  //   printf("A %i: (%i, %i)\n", i, V[i].x, V[i].y);

}

//----------------------------------------------------------------------------

// change coordinate systems from full image to cropped/scaled image (used for learning)

Point full2cropped(Point & v)
{
  Point vprime;

  float scale_factor = (float) output_width / (float) output_crop_width; 
  int x_offset = CANONICAL_IMAGE_WIDTH / 2 - output_crop_width / 2;
  int y_offset = output_crop_top_y;

  vprime.x = (float) (v.x - x_offset) * scale_factor;
  vprime.y = (float) (v.y - y_offset) * scale_factor;

  return vprime;
}

//----------------------------------------------------------------------------

// opposite of above

Point cropped2full(Point & v)
{
  Point vprime;

  float inv_scale_factor = (float) output_crop_width / (float) output_width; 
  int x_offset = CANONICAL_IMAGE_WIDTH / 2 - output_crop_width / 2;
  int y_offset = output_crop_top_y;

  vprime.x = x_offset + (float) v.x * inv_scale_factor;
  vprime.y = y_offset + (float) v.y * inv_scale_factor;

  return vprime;
}

//----------------------------------------------------------------------------


void dyn_draw_overlay()
{
  stringstream ss;
  int r, g, b;

  // which image is this?

  ss << current_index << ", " << dyn_current_seq_idx << ": " << current_imsig;
  string str = ss.str();

  putText(draw_im, str, Point(5, 10), FONT_HERSHEY_SIMPLEX, fontScale, Scalar::all(255), 1, 8);

  // horizontal lines for trail edge rows

  for (int i = 0; i < trailEdgeRow.size(); i++) 
    line(draw_im, Point(0, trailEdgeRow[i]), Point(draw_im.cols - 1, trailEdgeRow[i]), Scalar(0, 128, 128), 1);

  // verts for this image

  float inv_scale_factor = (float) output_crop_width / (float) output_width; 
  int x_offset = draw_im.cols / 2 - output_crop_width / 2;
  int y_offset = output_crop_top_y;

  if (!do_full_dyn) {

    // in clipped/scaled coord system
    
    if (dyn_current_seq_idx == 0) {
      
      line(draw_im, cropped2full(DynVert[current_index][dyn_current_seq_idx][0]), cropped2full(DynVert[current_index][dyn_current_seq_idx][1]), Scalar(0, 0, 255), 2);
      line(draw_im, cropped2full(DynVert[current_index][dyn_current_seq_idx][2]), cropped2full(DynVert[current_index][dyn_current_seq_idx][3]), Scalar(0, 0, 255), 2);
      
    }
    
    // in image coord system -- remember to convert before writing to file
    
    else {
      
      if (dyn_current_seq_idx == 3) {
	
	line(draw_im, cropped2full(DynVert[current_index][0][0]), cropped2full(DynVert[current_index][0][1]), Scalar(0, 0, 128), 1);
	line(draw_im, cropped2full(DynVert[current_index][0][2]), cropped2full(DynVert[current_index][0][3]), Scalar(0, 0, 128), 1);
      }
      
      if (dyn_current_seq_idx == 1) {
	g = b = 255;
	r = 0;
      }
      else if (dyn_current_seq_idx == 2) {
	r = b = 255;
	g = 0;
      }
      else if (dyn_current_seq_idx == 3) {
	r = b = 0;
	g = 255;
      }
      
      for (int i = 0; i < DynVert[current_index][dyn_current_seq_idx].size(); i += 2) {
	
	// only draw line segment if we have a PAIR of verts (this will NOT be the case while a new segment is being drawn) 
	
	if (i + 1 < DynVert[current_index][dyn_current_seq_idx].size())
	  line(draw_im, cropped2full(DynVert[current_index][dyn_current_seq_idx][i]), cropped2full(DynVert[current_index][dyn_current_seq_idx][i + 1]), Scalar(b, g, r), 2);
      }
    }
  }
  else {

    //   if (dyn_current_seq_idx == 0 || dyn_current_seq_idx == 10 || dyn_current_seq_idx == 20 || dyn_current_seq_idx == 30) {
      r = b = 0;
      g = 255;
      // }
      //    else {
      // r = b = 255;
      //g = 0;
      // }

    for (int i = 0; i < DynVert[current_index][dyn_current_seq_idx].size(); i += 2) {
	
      // only draw line segment if we have a PAIR of verts (this will NOT be the case while a new segment is being drawn) 
      
      if (i + 1 < DynVert[current_index][dyn_current_seq_idx].size())
	line(draw_im, cropped2full(DynVert[current_index][dyn_current_seq_idx][i]), cropped2full(DynVert[current_index][dyn_current_seq_idx][i + 1]), Scalar(b, g, r), 2);
    }

  }
}

//----------------------------------------------------------------------------

// what to do when a key is pressed

void dyn_onKeyPress(char c, bool print_help)
{
  int idx;
  int step_idx = 1;

  // goto image 0 in sequence
  
  if (c == '0' || print_help) {

    if (print_help) 
      printf("0 = return to index 0 image\n");
    else
      current_index = 0;

  }

  // goto next image 
  
  if (c == 'x' || c == 'X' || print_help) {

    if (print_help)
      printf("x = goto next image, X = take bigger step to next image\n");
    else if (!dragging) {
      if (c == 'X')
	step_idx = BIG_INDEX_STEP;

      current_index += step_idx;

      if (current_index >= DynVert.size())
	current_index -= DynVert.size();
    }
  }
  
  // goto previous image 
  
  if (c == 'z' || c == 'Z' || print_help) {
    
    if (print_help) 
      printf("z = goto previous image, Z = take bigger step to previous image\n");
    else if (!dragging) {
      if (c == 'Z')
	step_idx = BIG_INDEX_STEP;

      current_index -= step_idx;

      if (current_index < 0)
	current_index = DynVert.size() + current_index;
    }
  }

  // same image, next in sequence
  
  if (c == 's' || print_help) {

    if (print_help)
      printf("s = same image, next in SEQUENCE\n");
    else {

      dyn_current_seq_idx++;

      if (do_full_dyn) {
	if (dyn_current_seq_idx >= FULL_DYN_SEQ_LENGTH)
	  dyn_current_seq_idx -= FULL_DYN_SEQ_LENGTH;
      }
      else {
	if (dyn_current_seq_idx >= DYN_SEQ_LENGTH)
	  dyn_current_seq_idx -= DYN_SEQ_LENGTH;
      }
    }
  }
  
  // same image, previous in sequence
  
  if (c == 'a' || print_help) {
    
    if (print_help) 
      printf("a = same image, previous in SEQUENCE\n");
    else {

      dyn_current_seq_idx--;

      if (dyn_current_seq_idx < 0) {
	if (do_full_dyn)
	  dyn_current_seq_idx = FULL_DYN_SEQ_LENGTH + dyn_current_seq_idx;
	else
	  dyn_current_seq_idx = DYN_SEQ_LENGTH + dyn_current_seq_idx;
      }
    }
  }

  // save everything

  if (c == 'S' || print_help) {
    if (print_help)
      printf("s = save verts\n");
    else {
      //      printf("saving disabled\n");
      dyn_save_vertfile(1);
      dyn_save_vertfile(2);
      dyn_save_vertfile(3);
     
      //      exit(1);
    }
  }
  
  /*
  // toggle showing crop rectangle

  if (c == 'c' || print_help) {
    if (print_help)
      printf("c = toggle show crop rect\n");
    else
      do_show_crop_rect = !do_show_crop_rect;
  }

  // toggle overlay
  
  if (c == 'o' || print_help) {
    if (print_help)
      printf("o = toggle show overlay\n");
    else
      do_overlay = !do_overlay;
  }

  // toggle randomized index mode
  
  if (c == 'r' || print_help) {
    if (print_help)
      printf("r = toggle randomized index mode\n");
    else {
      if (!do_verts)
	do_random = !do_random;
    }
  }

  // toggle vert image-only mode
  
  if (c == 'v' || print_help) {
    if (print_help) 
      printf("v = toggle vert image only mode\n");
    else {
      if (vert_current_index) {
	do_verts = !do_verts;
	if (do_verts)
	  do_random = false;
      }
    }
  }

  // write traintest

  if (c == 'T' || print_help) {
    if (print_help)
      printf("T = write traintest for localization\n");
    else {
      //    write_traintest_vanilla_grayscale();
#ifdef OUTPUT_CHOCOLATE_DATA
      write_traintest_chocolate_color();
#elif defined(OUTPUT_STRAWBERRY_DATA)
      write_traintest_strawberry_color();
#elif defined(OUTPUT_PEACH_DATA)
      write_traintest_peach_color();
#else
      printf("undefined output data type\n");
      exit(1);
#endif
    }
  }

  // write non-trail

  if (c == 'N' || print_help) {
    if (print_help)
      printf("N = write traintest for trail/not-trail\n");
    else 
      write_traintest_nontrail_color();
  }
  */
}

//----------------------------------------------------------------------------

// what to do when mouse is moved/mouse button is push/released

void dyn_onMouse(int event, int x, int y, int flags, void *userdata)
{
  Point v, vcropped;
  int g, b;

  // no mouse interaction if this is seq_idx 0

  if (dyn_current_seq_idx == 0)
    return;

  // dragging...

  if  ( event == EVENT_MOUSEMOVE ) {

    //    vx = x;
    v.x = x;
    if (dragging) {
      g = 200;
      b = 0;
      v.y = dragging_y;
    }
    else {
      g = 255; 
      b = 255;
      v.y = snap_y(y);
    }

    //    printf("DRAGGING %i %i\n", v.x, v.y); fflush(stdout);
  }

  // initiating horizontal segment

  else if  ( event == EVENT_LBUTTONDOWN ) {

    dragging = true;

    g = 255;
    b = 0;
    v.x = x;
    v.y = snap_y(y);
    dragging_x = v.x;
    dragging_y = v.y;

    // clear any existing verts *on this row*

    vcropped = full2cropped(v);

    VertVect::iterator iter = DynVert[current_index][dyn_current_seq_idx].begin();

    while (iter != DynVert[current_index][dyn_current_seq_idx].end()) {
      if (fabs((*iter).y - vcropped.y) <= Y_TOLERANCE)
	iter = DynVert[current_index][dyn_current_seq_idx].erase(iter);
      else
	iter++;
    }

    //    printf("LBUTTON DOWN %i %i\n", v.x, v.y); fflush(stdout);
    DynVert[current_index][dyn_current_seq_idx].push_back(vcropped);
  }

  // finishing horizontal segment

  else if  ( event == EVENT_LBUTTONUP ) {

    dragging = false;

    g = 255;
    b = 0;
    v.x = x;
    v.y = dragging_y;

    //    printf("LBUTTON UP %i %i\n", v.x, v.y); fflush(stdout);

    DynVert[current_index][dyn_current_seq_idx].push_back(full2cropped(v));

    // done with this image!

    if (DynVert[current_index][dyn_current_seq_idx].size() == REQUIRED_NUMBER_OF_VERTS_PER_IMAGE) {

      sort_vertices(DynVert[current_index][dyn_current_seq_idx]);

      //Vert_idx_set.insert(current_index);

      //      set<int>::iterator iter = NoVert_idx_set.find(current_index);
      // // if it is actually in the set, erase it
      //if (iter != NoVert_idx_set.end())
      //	NoVert_idx_set.erase(iter);

      vert_current_index = true;

      //      if (next_nonvert_idx != NO_INDEX)
      //	next_nonvert_idx = most_isolated_nonvert_image_idx(current_index);
    }

  }

  // clear vertices for this image

  else if  ( event == EVENT_RBUTTONDOWN ) {

    erasing = true;

  }

  // clear vertices for this image

  else if  ( event == EVENT_RBUTTONUP ) {

    erasing = false;

    DynVert[current_index][dyn_current_seq_idx].clear();

    /*
    set<int>::iterator iter = Vert_idx_set.find(current_index);
    // if it is actually in the set, erase it
    if (iter != Vert_idx_set.end())
      Vert_idx_set.erase(iter);
    */

    //    NoVert_idx_set.insert(current_index);

    vert_current_index = false;

    // this could be done more efficiently, but it should be a pretty rare event
    //if (next_nonvert_idx != NO_INDEX)
    //  next_nonvert_idx = most_isolated_nonvert_image_idx();
  }

  draw_im = current_im.clone();
  dyn_draw_overlay();

  // show current edit that is underway

  if (dragging)
    line(draw_im, Point(dragging_x, v.y), Point(v.x, v.y), Scalar(0, 255, 255), 2);

  // where is cursor?

  if (!erasing)
    circle(draw_im, Point (v.x, v.y), 8, Scalar(0, g, b), 1, 8, 0);

  imshow("trailGT", draw_im);  
}

//----------------------------------------------------------------------------

void dyn_save_vertfile(int seq_idx)
{
  int num_saved = 0;
  stringstream ss;

  ss << "_" << seq_idx << ".txt";
  string filename = dyn_filename_stem + ss.str();

  //  printf("%s\n", filename.c_str());

  FILE *fp = fopen(filename.c_str(), "w");

  for (int i = 0; i < DynVert.size(); i++) {
    if (DynVert[i][seq_idx].size() == REQUIRED_NUMBER_OF_VERTS_PER_IMAGE) {
      num_saved++;
      fprintf(fp, "%s, %s, ", DynShortname[i].c_str(), fullpathname_to_signature(DynFullpathname[i][seq_idx]).c_str());
      for (int j = 0; j < DynVert[i][seq_idx].size() - 1; j++)
	fprintf(fp, "%i, ", (int) rint(DynVert[i][seq_idx][j].x));
      fprintf(fp, "%i\n", DynVert[i][seq_idx][DynVert[i][seq_idx].size() - 1].x);
    }
  }

  fclose(fp);

  printf("%i: saved %i\n", seq_idx, num_saved);
}

//----------------------------------------------------------------------------

// make sure vertices are all there and sorted, etc.

bool check_verts_good(VertVect & V)
{
  if (V.size() != 4)
    return false;

  if (V[0].y != 26 || V[1].y != 26 || V[2].y != 74 || V[2].y != 74)
    return false;

  if (V[0].x > V[1].x || V[2].x > V[3].x)
    return false;

  return true;
}

//----------------------------------------------------------------------------

// V3 = t1 * V1 + (1 - t1) * V2

void interpolate_verts(float t1, VertVect & V1, VertVect & V2, VertVect & V3)
{
  Point v;

  V3.clear();

  for (int i = 0; i < V1.size(); i++) {
    v.x = t1 * V1[i].x + (1.0 - t1) * V2[i].x;
    v.y = t1 * V1[i].y + (1.0 - t1) * V2[i].y;
    V3.push_back(v);
  }

}

//----------------------------------------------------------------------------

// 
// X is 4 x 1 holding (a, b, c, d)

double evaluate_cubic(double t, Mat & X)
{
  Mat A(1, 4, CV_32FC1);

  A.at<float>(0, 0) = t*t*t;
  A.at<float>(0, 1) = t*t;
  A.at<float>(0, 2) = t;
  A.at<float>(0, 3) = 1.0;

  Mat Y = A * X;

  return Y.at<float>(0, 0);

  // put t into A = 1 x 4 = [t^3   t^2  t   1]
  // multiply A * X = Y
  // read off Y.at<float>(0, 0)
}

//----------------------------------------------------------------------------

// VV is 4 images
// VV[i] has 4 vertices

// VV[0][0], VV[1][0], VV[2][0], VV[3][0] are upper-left verts

// from http://www.had2know.com/academics/cubic-through-4-points.html

void cubic_interpolation(int index)
//vector < VertVect > & VV)
{
  vector < VertVect> VV = DynVert[index];
  
  int i, j;

  vector <Mat> X;
  Mat Y(4, 1, CV_32FC1);
  Mat A(4, 4, CV_32FC1);
  Mat Y_prime;

  A.at<float>(0, 0) = 0.0;
  A.at<float>(0, 1) = 0.0;
  A.at<float>(0, 2) = 0.0;
  A.at<float>(0, 3) = 1.0;

  A.at<float>(1, 0) = 1.0;
  A.at<float>(1, 1) = 1.0;
  A.at<float>(1, 2) = 1.0;
  A.at<float>(1, 3) = 1.0;

  A.at<float>(2, 0) = 8.0;
  A.at<float>(2, 1) = 4.0;
  A.at<float>(2, 2) = 2.0;
  A.at<float>(2, 3) = 1.0;

  A.at<float>(3, 0) = 27.0;
  A.at<float>(3, 1) = 9.0;
  A.at<float>(3, 2) = 3.0;
  A.at<float>(3, 3) = 1.0;

  cout << "A = " << A << endl;

  Mat A_inv;
  invert(A, A_inv);

  cout << "A_inv = " << A_inv << endl;

  //  printf("size %i\n", VV.size());

  for (j = 0; j < REQUIRED_NUMBER_OF_VERTS_PER_IMAGE; j++) {
    for (i = 0; i < VV.size(); i++) {
      //      printf("seq %i, vert %i: %i, %i\n", i, j, VV[i][j].x, VV[i][j].y); 
      Y.at<float>(i, 0) = VV[i][j].x;
    }
    printf("Y %i\n", j);
    cout << Y << endl; 
    X.push_back(A_inv * Y);
    //    printf("X %i\n", j);
    //    cout << X << endl;
    //    Y_prime = A * X;
    //    printf("Y_prime %i\n", j);
    //    cout << Y_prime << endl;
  }


  for (int seq_index = 0; seq_index < FULL_DYN_SEQ_LENGTH; seq_index++) {
    float t = 0.1 * (float) seq_index;
    //    printf("%03i\n", seq_index);

    string logdir = logdir_from_signature(DynSig[index]);
    string seqfullpathname = fullpathname_from_logdir_and_seqnum(logdir,
								 seqnum_from_signature(DynSig[index]) - seq_index);

    fprintf(DynOutFP[seq_index], "%s, %s, ", DynShortname[index].c_str(), fullpathname_to_signature(seqfullpathname).c_str());
    for (j = 0; j < REQUIRED_NUMBER_OF_VERTS_PER_IMAGE; j++) {
      if (j == REQUIRED_NUMBER_OF_VERTS_PER_IMAGE - 1)
	fprintf(DynOutFP[seq_index], "%.2lf\n", evaluate_cubic(t, X[j]));
      else
	fprintf(DynOutFP[seq_index], "%.2lf, ", evaluate_cubic(t, X[j]));

      //    for (float t = 0.0; t <= 3.01; t += 0.1)
      //      printf("%i: f(%03i) = %.2lf\n", j, index, evaluate_cubic(t, X[j]));
    }
    printf("\n");
  }

  printf("\n");
}

//----------------------------------------------------------------------------

void dyn_load_distortfile(string & distortfilename)
{
  ifstream inStream;
  string line;

  // read file

  inStream.open(distortfilename.c_str());

  if (!inStream.good()) {
    printf("problem opening %s -- exiting\n", distortfilename.c_str());
    exit(1);
  }

  DistortParams D;

  while (getline(inStream, line)) {

    istringstream iss(line);
    string short_ss, imsig, dist1_ss, dist2_ss, dist3_ss, dist4_ss, dist5_ss;

    iss >> short_ss >> imsig >> dist1_ss >> dist2_ss >> dist3_ss >> dist4_ss >> dist5_ss;

    // D.x_left_new, D.x_right_new, D.do_horizontal_flip, D.contrast_factor, D.brightness_factor
    //  int x_left_new, x_right_new;                // encodes horizontal shift, scaling
    //bool do_horizontal_flip;                    // do flip?
    //float contrast_factor, brightness_factor;   // encodes value scaling, shift of each pixel channel

    short_ss.erase(short_ss.end() - 1);
    imsig.erase(imsig.end() - 1);
    dist1_ss.erase(dist1_ss.end() - 1);
    dist2_ss.erase(dist2_ss.end() - 1);
    dist3_ss.erase(dist3_ss.end() - 1);
    dist4_ss.erase(dist4_ss.end() - 1);
    dist5_ss.erase(dist5_ss.end() - 1);

    D.x_left_new = atoi(dist1_ss.c_str());
    D.x_right_new = atoi(dist2_ss.c_str());
    D.do_horizontal_flip = atoi(dist3_ss.c_str());
    D.contrast_factor = atof(dist4_ss.c_str());
    D.brightness_factor = atof(dist5_ss.c_str());

    //    printf("%s %s %s %s %s %s %s\n", short_ss.c_str(), imsig.c_str(), dist1_ss.c_str(), dist2_ss.c_str(), dist3_ss.c_str(), dist4_ss.c_str(), dist5_ss.c_str());
    //printf("%s %s\n", short_ss.c_str(), imsig.c_str());
    //    print_distortion(stdout, D);

    // add short_ss, imsig, D to vectors

    DistortDynShortname.push_back(short_ss);
    DistortDynSig.push_back(imsig);
    DistortDynParams.push_back(D);

  }

  //  exit(1);
}

//----------------------------------------------------------------------------

// seq_idx will be 0, ..., DYN_SEQ_LENGTH - 1

void dyn_load_vertfile(string & vertfilename, int seq_idx)
{
  ifstream inStream;
  string line;
  string fullpath_prefix("/warthog_logs/");
  Mat im, seqim;
  string fullpathname;
  Point v, vfarfull, vfarcropped, vnearfull, vnearcropped;
  int image_idx;

  // read file

  inStream.open(vertfilename.c_str());

  if (!inStream.good()) {
    printf("problem opening %s -- exiting\n", vertfilename.c_str());
    exit(1);
  }

  image_idx = 0;

  vfarfull.y = IMAGE_ROW_FAR;
  vnearfull.y = IMAGE_ROW_NEAR;
  vfarcropped = full2cropped(vfarfull);
  vnearcropped = full2cropped(vnearfull);

  int num_loaded = 0;

  while (getline(inStream, line)) {
    //    printf("%s\n", line.c_str());

    vector <string> Seqfullpathname_vect;
    vector < VertVect > VV;  
    istringstream iss(line);
    string short_ss, imsig, far_x0_ss, far_x1_ss, near_x0_ss, near_x1_ss;

    iss >> short_ss >> imsig >> far_x0_ss >> far_x1_ss >> near_x0_ss >> near_x1_ss;

    // remove trailing commas
    short_ss.erase(short_ss.end() - 1);
    imsig.erase(imsig.end() - 1);
    far_x0_ss.erase(far_x0_ss.end() - 1);
    far_x1_ss.erase(far_x1_ss.end() - 1);
    near_x0_ss.erase(near_x0_ss.end() - 1);

    VertVect V;

    v.y = vfarcropped.y;  // IMAGE_ROW_FAR;
    v.x = atoi(far_x0_ss.c_str());
    V.push_back(v);
    v.x = atoi(far_x1_ss.c_str());
    V.push_back(v);

    v.y = vnearcropped.y; // IMAGE_ROW_NEAR;
    v.x = atoi(near_x0_ss.c_str());
    V.push_back(v);
    v.x = atoi(near_x1_ss.c_str());
    V.push_back(v);

    fullpathname = fullpath_prefix + imsig;

    int index = seqnum_from_signature(imsig);
    string logdir = logdir_from_signature(imsig);

    //    printf("%i: %s %s\n", image_idx, short_ss.c_str(), imsig.c_str());

    if (seq_idx == 0) {

      DynShortname.push_back(short_ss);
      DynSig.push_back(imsig);

      VV.resize(DYN_SEQ_LENGTH);
      DynVert.push_back(VV);                             

      for (int i = 0; i < DYN_SEQ_LENGTH; i++) {
	string seqfullpathname = fullpathname_from_logdir_and_seqnum(logdir, index - i * DYN_SEQ_INTERVAL);
	Seqfullpathname_vect.push_back(seqfullpathname);
	//	printf("%s\n", seqfullpathname.c_str());
      }

      DynFullpathname.push_back(Seqfullpathname_vect);    // has names for all images in sequence

      for (int j = 1; j < DYN_SEQ_LENGTH; j++)
	DynVert[image_idx][j].clear();

      DynVert[image_idx][seq_idx] = V;
      //saveVertVect(stdout, DynVert[image_idx][seq_idx]);
      
      image_idx++;

      num_loaded++;
    }
    else {

      // figure out image_idx from short_ss

      //      printf("%s\n", short_ss.c_str());
      string index_str = short_ss.substr(short_ss.length() - NUM_SEQ_DIGITS, NUM_SEQ_DIGITS); 
      image_idx = atoi(index_str.c_str());
      //      printf("%s -> %s -> %i\n", short_ss.c_str(), index_str.c_str(), image_idx);

      DynVert[image_idx][seq_idx] = V;
      num_loaded++;

    }

  }
 
  printf("%i: %i loaded\n", seq_idx, num_loaded);

    //    add_images(line, Fullpathname_vect);

  /*
  if (seq_idx == 3) {

    char *outstr = (char *) malloc(256*sizeof(char));

    for (int full_seq_index = 0; full_seq_index < FULL_DYN_SEQ_LENGTH; full_seq_index++) {
      //      sprintf(outstr, "dyn_trainvert_%03i.txt", full_seq_index);
      sprintf(outstr, "dyn_testvert_%03i.txt", full_seq_index);
      //      DynOutFilename.push_back(string(outstr));
      DynOutFP.push_back(fopen(outstr, "w"));
    }

    //    for (int full_seq_index = 0; full_seq_index < FULL_DYN_SEQ_LENGTH; full_seq_index++) {
    //  cout << DynOutFilename[full_seq_index] << endl;
    // }

    for (int i = 0; i < DynVert.size(); i++) {

      if (!check_verts_good(DynVert[i][0]) || !check_verts_good(DynVert[i][1]) ||
	  !check_verts_good(DynVert[i][2]) || !check_verts_good(DynVert[i][3]))
	printf("bad %i\n", i);
 
      //      interpolate_verts(0.5, DynVert[i][0], DynVert[i][2], DynVert[i][1]);
      //interpolate_verts(0.66667, DynVert[i][0], DynVert[i][3], DynVert[i][1]);
      //      interpolate_verts(0.33333, DynVert[i][0], DynVert[i][3], DynVert[i][2]);

      cubic_interpolation(i);
    
    }

    for (int full_seq_index = 0; full_seq_index < FULL_DYN_SEQ_LENGTH; full_seq_index++) 
      fclose(DynOutFP[full_seq_index]);

    exit(1);
  }
  */
}

//----------------------------------------------------------------------------

// seq_idx will be 0, ..., 30 (FULL_DYN_SEQ_LENGTH - 1)

void full_dyn_load_vertfile(string & vertfilename, int seq_idx)
{
  ifstream inStream;
  string line;
  string fullpath_prefix("/warthog_logs/");
  Mat im, seqim;
  string fullpathname;
  Point v, vfarfull, vfarcropped, vnearfull, vnearcropped;
  int image_idx;

  // read file

  inStream.open(vertfilename.c_str());

  if (!inStream.good()) {
    printf("problem opening %s -- exiting\n", vertfilename.c_str());
    exit(1);
  }

  image_idx = 0;

  vfarfull.y = IMAGE_ROW_FAR;
  vnearfull.y = IMAGE_ROW_NEAR;
  vfarcropped = full2cropped(vfarfull);
  vnearcropped = full2cropped(vnearfull);

  int num_loaded = 0;

  while (getline(inStream, line)) {
    //    printf("%s\n", line.c_str());

    vector <string> Seqfullpathname_vect;
    vector < VertVect > VV;  
    istringstream iss(line);
    string short_ss, imsig, far_x0_ss, far_x1_ss, near_x0_ss, near_x1_ss;

    iss >> short_ss >> imsig >> far_x0_ss >> far_x1_ss >> near_x0_ss >> near_x1_ss;

    // remove trailing commas
    short_ss.erase(short_ss.end() - 1);
    imsig.erase(imsig.end() - 1);
    far_x0_ss.erase(far_x0_ss.end() - 1);
    far_x1_ss.erase(far_x1_ss.end() - 1);
    near_x0_ss.erase(near_x0_ss.end() - 1);

    VertVect V;

    v.y = vfarcropped.y;  // IMAGE_ROW_FAR;
    v.x = atof(far_x0_ss.c_str());
    V.push_back(v);
    v.x = atof(far_x1_ss.c_str());
    V.push_back(v);

    v.y = vnearcropped.y; // IMAGE_ROW_NEAR;
    v.x = atof(near_x0_ss.c_str());
    V.push_back(v);
    v.x = atof(near_x1_ss.c_str());
    V.push_back(v);

    fullpathname = fullpath_prefix + imsig;

    int index = seqnum_from_signature(imsig);
    string logdir = logdir_from_signature(imsig);

    //    printf("%i: %s %s\n", image_idx, short_ss.c_str(), imsig.c_str());

    if (seq_idx == 0) {

      DynShortname.push_back(short_ss);
      DynSig.push_back(imsig);

      VV.resize(FULL_DYN_SEQ_LENGTH);
      DynVert.push_back(VV);                             

      for (int i = 0; i < FULL_DYN_SEQ_LENGTH; i++) {
	string seqfullpathname = fullpathname_from_logdir_and_seqnum(logdir, index - i);
	Seqfullpathname_vect.push_back(seqfullpathname);
	//	printf("%s\n", seqfullpathname.c_str());
      }

      DynFullpathname.push_back(Seqfullpathname_vect);    // has names for all images in sequence

      for (int j = 1; j < FULL_DYN_SEQ_LENGTH; j++)
	DynVert[image_idx][j].clear();

      DynVert[image_idx][seq_idx] = V;
      //saveVertVect(stdout, DynVert[image_idx][seq_idx]);
      
      image_idx++;

      num_loaded++;
    }
    else {

      // figure out image_idx from short_ss

      //      printf("%s\n", short_ss.c_str());
      string index_str = short_ss.substr(short_ss.length() - NUM_SEQ_DIGITS, NUM_SEQ_DIGITS); 
      image_idx = atoi(index_str.c_str());
      //      printf("%s -> %s -> %i\n", short_ss.c_str(), index_str.c_str(), image_idx);

      DynVert[image_idx][seq_idx] = V;
      num_loaded++;

    }

  }
 
  //  printf("%i: %i loaded\n", seq_idx, num_loaded);
}

//----------------------------------------------------------------------------

// 0, 1, 2, and 3

void annotate_dynamics(bool do_train)
{
  if (do_train)
    dyn_filename_stem = string("dyn_trainvert");
  else
    dyn_filename_stem = string("dyn_testvert");

  string filename = dyn_filename_stem + string(".txt");
  dyn_load_vertfile(filename, 0);

  filename = dyn_filename_stem + string("_1.txt");
  dyn_load_vertfile(filename, 1);

  filename = dyn_filename_stem + string("_2.txt");
  //filename = string("bak2.dyn_trainvert_2.txt");
  dyn_load_vertfile(filename, 2);

  filename = dyn_filename_stem + string("_3.txt");
  dyn_load_vertfile(filename, 3);

  current_index = 0;
  dyn_current_seq_idx = 0;

  // display

  char c;

  do {
    
    // load image

    current_imname = DynFullpathname[current_index][dyn_current_seq_idx];
    current_imsig = fullpathname_to_signature(current_imname);

    current_im = smart_imread(current_imname);
    draw_im = current_im.clone();

    // show image 

    dyn_draw_overlay();
    //    draw_other_windows();
    imshow("trailGT", draw_im);
    if (!callbacks_set) {
      setMouseCallback("trailGT", dyn_onMouse);
      callbacks_set = true;
    }

    c = waitKey(0);

    dyn_onKeyPress(c);

  } while (c != (int) 'q');

  exit(1);
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

// 0, ..., 30 (FULL_DYN_SEQ_LENGTH - 1)

void full_annotate_dynamics(bool do_train)
{
  do_full_dyn = true;

  // this is 16140

  if (do_train) {
    dyn_filename_stem = string("dyn_trainvert");
    string distortfilename = string("distorted_traindistorts.txt");
    dyn_load_distortfile(distortfilename);
  }
  else
    dyn_filename_stem = string("dyn_testvert");

  // this is 1614 x 30

  char *filename = (char *) malloc(512*sizeof(char));
  for (int full_seq_index = 0; full_seq_index < FULL_DYN_SEQ_LENGTH; full_seq_index++) {
    sprintf(filename, "%s_%03i.txt", dyn_filename_stem.c_str(), full_seq_index);
    string s(filename);
    full_dyn_load_vertfile(s, full_seq_index);
    //    printf("%s\n", filename);
  }

  /*
  // apply distortions to every GT in sequence to make distorted_trainvert_000.txt, ..., distorted_trainvert_030.txt

  Point P;
  P = Point(0, IMAGE_ROW_FAR);
  Point p_far = full2cropped(P);
  P = Point(0, IMAGE_ROW_NEAR);
  Point p_near = full2cropped(P);

  // looping over 16140 distortions (10 for each training image)

  stringstream line_ss;
  vector < vector <string> > Line_distorted;
  Line_distorted.resize(DistortDynParams.size());

  for (int i = 0; i < DistortDynParams.size(); i++) {    // 16K+
  //  for (int i = 0; i < 10; i++) {    // 16K+

    int t;
    bool sigfound = false;

    // look for signature DistortDynSig[i] among 1614 in DynSig, use its index there for DynVert -- "i" is WRONG

    for (t = 0; t < DynSig.size(); t++) 
      if (DistortDynSig[i] == DynSig[t]) {
	sigfound = true;
	break;
      }

    if (sigfound)
      printf("found match %i: %s and %s\n", t, DistortDynSig[i].c_str(), DynSig[t].c_str());
    else {
      printf("no match\n");
      exit(1);
    }

    int index = seqnum_from_signature(DistortDynSig[i]);
    string logdir = logdir_from_signature(DistortDynSig[i]);

    Line_distorted[i].resize(FULL_DYN_SEQ_LENGTH);

    for (int j = FULL_DYN_SEQ_LENGTH - 1; j >= 0; j--) {      // 0-30

      string seqfullpathname = fullpathname_from_logdir_and_seqnum(logdir, index - j);

      //      printf("%i, %i, %i: %s\n", i, t, j, seqfullpathname.c_str());

      vector <int> FarXval, NearXval;
      vector <int> FarXval_new, NearXval_new;

      FarXval.push_back(DynVert[t][j][0].x);
      FarXval.push_back(DynVert[t][j][1].x);

      NearXval.push_back(DynVert[t][j][2].x);
      NearXval.push_back(DynVert[t][j][3].x);

      FarXval_new.resize(FarXval.size());
      NearXval_new.resize(NearXval.size());

      Mat seqim = smart_imread(seqfullpathname);
      
      Mat distorted_output_im = apply_precomputed_distortion(DistortDynParams[i],
							     seqim, 
							     FarXval, FarXval_new,
							     NearXval, NearXval_new);

      //      printf("%i x %i\n", distorted_output_im.cols, distorted_output_im.rows);
      //      print_distortion(stdout, DistortDynParams[i]);
      //      printf("%i %i %i %i\n", FarXval[0], FarXval[1], NearXval[0], NearXval[1]);
      //      printf("%i %i %i %i\n", FarXval_new[0], FarXval_new[1], NearXval_new[0], NearXval_new[1]);

      // draw overlay

      Point p_far_left = Point(FarXval_new[0], p_far.y);
      Point p_far_right = Point(FarXval_new[1], p_far.y);
      Point p_near_left = Point(NearXval_new[0], p_near.y);
      Point p_near_right = Point(NearXval_new[1], p_near.y);

      line_ss.str("");
      //      line_ss << short_ss.str().c_str() << ", " << imsig.c_str() << ", " << FarXval_new[0] << ", " << FarXval_new[1] << ", " << NearXval_new[0] << ", " << NearXval_new[1];
      line_ss << DistortDynShortname[i] << ", " << fullpathname_to_signature(seqfullpathname) << ", " << FarXval_new[0] << ", " << FarXval_new[1] << ", " << NearXval_new[0] << ", " << NearXval_new[1];
      Line_distorted[i][j] = line_ss.str();

      if (j == 0) {
	printf("%s\n", Line_distorted[i][j].c_str());

	line(distorted_output_im, p_far_left, p_far_right, Scalar(0, 255, 0), 2);
	line(distorted_output_im, p_near_left, p_near_right, Scalar(0, 255, 0), 2);

	imshow("mm", distorted_output_im);
	//imshow("mm", seqim);
	char c = waitKey(5);

	if (c == 'q')
	  exit(1);
      }

      //exit(1);

    }
  }
  
  
  vector <FILE *> Distorted_train_vert_fp;

  for (int j = 0; j < Line_distorted[0].size(); j++) {
    sprintf(filename, "distorted_trainvert_%03i.txt", j);
    //    printf("%s\n", filename);
    Distorted_train_vert_fp.push_back(fopen(filename, "w"));
  }

  for (int i = 0; i < Line_distorted.size(); i++) {
    //for (int i = 0; i < 10; i++) {
    for (int j = 0; j < Line_distorted[i].size(); j++) {
    fprintf(Distorted_train_vert_fp[j], "%s\n", Line_distorted[i][j].c_str());
    fflush(Distorted_train_vert_fp[j]);
    }
  }

  for (int j = 0; j < Line_distorted[0].size(); j++) 
    fclose(Distorted_train_vert_fp[j]);

  exit(1);
  */

  current_index = 0;
  dyn_current_seq_idx = 0;

  // display

  char c;

  do {
    
    // load image

    current_imname = DynFullpathname[current_index][dyn_current_seq_idx];
    current_imsig = fullpathname_to_signature(current_imname);

    current_im = smart_imread(current_imname);
    draw_im = current_im.clone();

    // show image 

    dyn_draw_overlay();
    //    draw_other_windows();
    imshow("trailGT", draw_im);
    if (!callbacks_set) {
      setMouseCallback("trailGT", dyn_onMouse);
      callbacks_set = true;
    }

    c = waitKey(0);

    dyn_onKeyPress(c);

  } while (c != (int) 'q');

  exit(1);
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

void compute_trajectory()
{
  int total = loadVertMap();

  printf("compute_traj(): %i\n", total);

  printf("%i, %i\n", (int) Vert_idx_set.size(), (int) External_vert_vect.size());
  fflush(stdout);

  string sig;
  VertVect V;
  int traj_number = -1;
  string logdir;
  string last_logdir("");
  FILE *fp = fopen("traj.txt", "w");
  vector <int> x_far;
  vector <int> x_near;

  for (int i = 0; i < External_vert_vect.size(); i++) {
    sig = External_vert_vect[i].first;
    V = External_vert_vect[i].second;
    logdir = logdir_from_signature(sig).c_str();
    if (logdir != last_logdir)
      traj_number++;
    fprintf(fp, "%i, %s, %i, %i, ", i, sig.c_str(), traj_number, seqnum_from_signature(sig));

    x_far.clear();
    x_near.clear();
    for (int j = 0; j < V.size(); j++) {
      if (V[j].y == IMAGE_ROW_FAR)
	x_far.push_back(V[j].x);
      else if (V[j].y == IMAGE_ROW_NEAR)
	x_near.push_back(V[j].x);
    }

    sort(x_far.begin(), x_far.end());
    sort(x_near.begin(), x_near.end());
    fprintf(fp, "%i, %i, %i, %i\n", x_far[0], x_far[1], x_near[0], x_near[1]);
    fflush(fp);
    
    //    saveVertVectNoParens(fp, V);

    last_logdir = logdir;
  }

  fclose(fp);
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

Mat smart_imread(string & fullpathname)
{
  Mat im = imread(fullpathname.c_str());

  // already cropped and rotated...proceed

  if (im.cols == CANONICAL_IMAGE_WIDTH)
    return im;

  // we have a "full" omni image, so first rotate & crop
  // specifically: throw away left half of im and rotate right half 90 degs counter-clockwise

  else if (im.cols == CANONICAL_IMAGE_HEIGHT * 2 && im.rows == CANONICAL_IMAGE_WIDTH) {
    
    Mat im_t = Mat(im.cols, im.rows, CV_8UC3);
    
    transpose(im, im_t);
    flip(im_t, im_t, 0);

    Rect crop_rect(0, 0, CANONICAL_IMAGE_WIDTH, CANONICAL_IMAGE_HEIGHT);
    Mat im_t_cropped = im_t(crop_rect);
    
    return im_t_cropped;
  }

  else {

    printf("image %s has unknown size -- punting\n", fullpathname.c_str());
    exit(1);

  }
}

//----------------------------------------------------------------------------

string fullpathname_from_logdir_and_seqnum(string & logdir, int seqnum)
{
  char *snum = (char *) malloc(10 * sizeof(char));
  sprintf(snum, "%06d.jpg", seqnum);
  string s = "/warthog_logs/" + logdir + "/omni_images/" + string(snum);

  free(snum);

  return s;
}

//----------------------------------------------------------------------------

string logdir_from_signature(string & sig)
{
  std::size_t pos;
  string s("");

  pos = sig.find("/");

  // we got it

  if (pos != std::string::npos) {
    
    // extract number substring starting at pos 

    s = sig.substr(0, pos);
    
  }

  return s;

}

//----------------------------------------------------------------------------

int seqnum_from_signature(string & sig)
{
  std::size_t pos;
  int seqnum = -1;

  pos = sig.rfind("/");

  // we got it

  if (pos != std::string::npos) {
    
    // extract number substring starting at pos 

    seqnum = atoi(sig.substr(pos + 1, NUM_SEQ_DIGITS).c_str());
    
  }

  return seqnum;
}

//----------------------------------------------------------------------------

// given something like /warthog_logs/Aug_05_2011_Fri_11_50_24_AM/omni_images/001000.jpg
// or /home/jiayi/prelim/data/Aug_05_2011_Fri_11_50_24_AM/omni_images/001000.jpg
// sets signature to Aug_05_2011_Fri_11_50_24_AM/omni_images/001000.jpg

// omni_images_right is also possible

string fullpathname_to_signature(string & fullpathname)
{
  set<string>::iterator iter;
  std::size_t pos;

  // find day of week -- it must be there or else error

  for (iter = dayofweek_set.begin(); iter != dayofweek_set.end(); iter++) {
    
    pos = fullpathname.rfind(*iter);

    // we got it

    if (pos != std::string::npos) {

      // extract substring starting at pos and running to end

      return fullpathname.substr(pos - MONTH_DAY_OFFSET, string::npos);

    }
  }

  // should not reach this spot

  return NULL;
}

//----------------------------------------------------------------------------

// smaller numbers -> MORE similar

float image_similarity(Mat & im1, Mat & im2)
{
// Calculate the L2 relative error between images
  double errorL2 = norm(im1, im2, CV_L2 );

  // Convert to a reasonable scale, since L2 error is summed across all pixels of the image
  double similarity = errorL2 / (double) ( im1.rows * im2.cols );

  return similarity;
}

//----------------------------------------------------------------------------


#define NUM_MNIST_CLASSES   10

void compute_mnist_classes(set <int> & idx_set, float x_min, float x_max)
{
  vector<int> hist;
  set<int>::iterator iter;
  int idx, i, bin;
  float x, range_width;
  int counter = 0;
  
  hist.resize(NUM_MNIST_CLASSES);
  for (i = 0; i < hist.size(); i++)
    hist[i] = 0;

  range_width = x_max - x_min;

  for (iter = idx_set.begin(); iter != idx_set.end(); iter++) {

    idx = *iter;

    //    for (int i = 0; i < Vert[idx].size(); i++) 
    //      printf("%i, %i: [%f, %f]\n", idx, i, (float) Vert[idx][i].x, (float) Vert[idx][i].y);

     
    for (i = 0; i < Vert[idx].size(); i += 2) {
      
      //      if (Vert[idx][i].y == IMAGE_ROW_NEAR && Vert[idx][i + 1].y == IMAGE_ROW_NEAR) {
      if (Vert[idx][i].y == IMAGE_ROW_FAR && Vert[idx][i + 1].y == IMAGE_ROW_FAR) {
	
	// middle of near trail row
	
	x = 0.5 * ((float) Vert[idx][i].x + (float) Vert[idx][i + 1].x);
	
	//	printf("%i [%i]: %f [%f, %f]\n", idx, i, x, (float) Vert[idx][i].x, (float) Vert[idx][i + 1].x);

	// clamp
	
	if (x < x_min)
	  x = x_min;
	else if (x > x_max)
	  x = x_max - 1;
	
	// bin
	
	bin = (int) floor((float) NUM_MNIST_CLASSES * (x - x_min) / range_width);
	hist[bin]++;
	
	//	printf("%i: %.2f -> %i\n", idx, x, bin);

	counter++;

      }
    }

    //    if (counter++ >= 10)
    //  break;
  }
  
    //  printf("NEAR min %i, max %i; FAR min %i, max %i [%i]\n", x_min_near, x_max_near, x_min_far, x_max_far);

  for (i = 0; i < hist.size(); i++)
    printf("%i: %.3f\n", i, (float) hist[i] / (float) counter);

}

//----------------------------------------------------------------------------

void calculate_gt_stats(set <int> & idx_set)
{
  set<int>::iterator iter;
  int n, x;
  int x_min_near, x_max_near;
  int x_min_far, x_max_far;
  
  n = 0;
  x_min_near = 10000;
  x_max_near = -10000;
  x_min_far = 10000;
  x_max_far = -10000;

  for (iter = idx_set.begin(); iter != idx_set.end(); iter++) {

    for (int i = 0; i < Vert[*iter].size(); i += 2) {
      x = (Vert[*iter][i].x + Vert[*iter][i + 1].x) / 2;
      if (Vert[*iter][i].y == IMAGE_ROW_NEAR && Vert[*iter][i + 1].y == IMAGE_ROW_NEAR) {
	if (x < x_min_near)
	  x_min_near = x;
	if (x > x_max_near)
	  x_max_near = x;
      }
      else {
	
	if (x < x_min_far)
	  x_min_far = x;
	if (x > x_max_far)
	  x_max_far = x;
      }
    }
    
    /*
    for (int i = 0; i < Vert[*iter].size(); i++) {

       if (Vert[*iter][i].y == IMAGE_ROW_NEAR) {

	 if (Vert[*iter][i].x < x_min_near)
	   x_min_near = Vert[*iter][i].x;
	 if (Vert[*iter][i].x > x_max_near)
	   x_max_near = Vert[*iter][i].x;
       }
       else {

	 if (Vert[*iter][i].x < x_min_far)
	   x_min_far = Vert[*iter][i].x;
	 if (Vert[*iter][i].x > x_max_far)
	   x_max_far = Vert[*iter][i].x;
       }

       n++;
    }
*/
  }

  //  printf("NEAR min %i, max %i; FAR min %i, max %i [%i]\n", x_min_near, x_max_near, x_min_far, x_max_far);
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

// this only does *BAD* (e.g., non-trail) images

void filter_out_overly_similar_bad_images()
{
  Mat im1, im2;
  float sim;
  set<int>::iterator iter, iter_next;
  
  printf("filtering out overly similar GT images\n");
  fflush(stdout);

  FilteredBad_idx_set.clear();

  for (iter = Bad_idx_set.begin(); iter != Bad_idx_set.end(); iter = iter_next) {

    im1 = smart_imread(Fullpathname_vect[*iter]);
    //    printf("%i\n", *iter);

    FilteredBad_idx_set.insert(*iter);

    iter_next = iter;
    iter_next++;

    // find next image that is dissimilar enough

    while (iter_next != Bad_idx_set.end()) {

      im2 = smart_imread(Fullpathname_vect[*iter_next]);

      sim = image_similarity(im1, im2);

      if (sim < SIMILARITY_THRESHOLD) {
	printf("skipping %i [%.3f with %i]\n", *iter_next, sim, *iter);
	//	printf("BAD: %i, %i: %.3f\n", *iter, *iter_next, sim);
      }
      else {
	//	printf("GOOD: %i, %i: %.3f\n", *iter, *iter_next, sim);
	break;
      }

      iter_next++;
    }
  }

  printf("%i after (%i before)\n", (int) FilteredBad_idx_set.size(), (int) Bad_idx_set.size());
  
}

//----------------------------------------------------------------------------

// this only does *VERT* images

void filter_out_overly_similar_images()
{
  Mat im1, im2;
  float sim;
  set<int>::iterator iter, iter_next;
  
  printf("filtering out overly similar GT images\n");
  fflush(stdout);

  FilteredVert_idx_set.clear();

  for (iter = Vert_idx_set.begin(); iter != Vert_idx_set.end(); iter = iter_next) {

    im1 = smart_imread(Fullpathname_vect[*iter]);
    //    printf("%i\n", *iter);

    FilteredVert_idx_set.insert(*iter);

    iter_next = iter;
    iter_next++;

    // find next image that is dissimilar enough

    while (iter_next != Vert_idx_set.end()) {

      im2 = smart_imread(Fullpathname_vect[*iter_next]);

      sim = image_similarity(im1, im2);

      if (sim < SIMILARITY_THRESHOLD) {
	printf("skipping %i [%.3f with %i]\n", *iter_next, sim, *iter);
	//	printf("BAD: %i, %i: %.3f\n", *iter, *iter_next, sim);
      }
      else {
	//	printf("GOOD: %i, %i: %.3f\n", *iter, *iter_next, sim);
	break;
      }

      iter_next++;
    }
  }

  printf("%i after (%i before)\n", (int) FilteredVert_idx_set.size(), (int) Vert_idx_set.size());
  
  //  compute_mnist_classes(FilteredVert_idx_set, 240- output_crop_width/2, 240+ output_crop_width/2);
  //  compute_mnist_classes(FilteredVert_idx_set, 200.0, 280.0);
    //calculate_gt_stats(FilteredVert_idx_set);

}

//----------------------------------------------------------------------------

// *starts* with Filtered and modifies it
 
void  filter_remove_far_edges_outside_crop()
{
  int i;
  set<int>::iterator iter, iter_next;
  bool badvert;
  int x_crop_left = CANONICAL_IMAGE_WIDTH / 2 - output_crop_width / 2;
  int x_crop_right = CANONICAL_IMAGE_WIDTH / 2 + output_crop_width / 2;

  printf("filtering verts outside crop region\n");
  fflush(stdout);

  iter = FilteredVert_idx_set.begin();

  while (iter != FilteredVert_idx_set.end()) {

    for (i = 0, badvert = false; i < Vert[*iter].size(); i++) {

      // bad?

      if (Vert[*iter][i].y == IMAGE_ROW_FAR && (Vert[*iter][i].x < x_crop_left || Vert[*iter][i].x >= x_crop_right)) {

	printf("%i: outside crop region\n", *iter);
	badvert = true;
	break;

      }
    }

    if (badvert) {
      iter_next = iter;
      iter_next++;
      FilteredVert_idx_set.erase(iter);
      iter = iter_next;
    }
    else 
      iter++;

  }
}

//----------------------------------------------------------------------------

// pick images that will actually be used for training/testing

void filter_for_traintest()
{
  filter_out_overly_similar_images();
  filter_remove_far_edges_outside_crop();

  printf("%i in train/test\n", (int) FilteredVert_idx_set.size());
  
}

//----------------------------------------------------------------------------

// erase member at iter in the middle of iterating

void leave_out_set_member(set<int>::iterator & iter, set<int>::iterator & iter_next, set <int> & idx_set)
{
  iter_next = iter;
  iter_next++;
  idx_set.erase(iter);
  iter = iter_next;
}

//----------------------------------------------------------------------------

// do not use GT images that have bad or missing images preceding them in sequence

// *starts* with passed argument idx_set (e.g., FilteredVert_idx_set or Vert_idx_set) and modifies it
 
void  filter_for_intact_sequence(set <int> & idx_set)
{
  int i, j;
  set<int>::iterator iter, iter_next, bad_iter;
  string fullpathname, sig, logdir;
  int index, index_offset;

  printf("filtering frames without complete preceding sequences\n");
  printf("before: %i\n", (int) idx_set.size());
  fflush(stdout);

  iter = idx_set.begin();

  // iterate over GT images

  while (iter != idx_set.end()) {

    fullpathname = Fullpathname_vect[*iter];
    sig = fullpathname_to_signature(fullpathname);
    index = seqnum_from_signature(sig);
    logdir = logdir_from_signature(sig);

    //    printf("%s, %s -> %i\n", fullpathname.c_str(), sig.c_str(), index);

    bool left_out = false;

    for (index_offset = 0; index_offset <= PEACH_SEQUENCE_LENGTH; index_offset++) {

      int seqnum = index - index_offset;
      string seqfullpathname = fullpathname_from_logdir_and_seqnum(logdir, seqnum);
      FILE *seqfp = fopen(seqfullpathname.c_str(), "r");

      // sequence image *exists*

      if (seqfp) {
	
	// don't actually need to do anything with it right now

	fclose(seqfp);

	// is sequence image already annotated somehow (either GT or bad)?

	string seqsig = fullpathname_to_signature(seqfullpathname);
	map<string, int>::iterator seqsig_iter = Signature_idx_map.find(seqsig);

	// yes, it is

	if (seqsig_iter != Signature_idx_map.end()) {

	  //	  printf("ANNO %i: %s (%i)\n", seqnum, seqfullpathname.c_str(), (*seqsig_iter).second);
	  
	  // is it "bad"?  

	  //	  bad_iter = Bad_idx_set.find((*sig_iter).second);

	  //	  if (bad_iter != Bad_idx_set.end()) {
	  if (isBad((*seqsig_iter).second)) {
	    printf("BAD %i: %s\n", seqnum, seqfullpathname.c_str());
	    leave_out_set_member(iter, iter_next, idx_set);
	    left_out = true;
	    break;
	  }
	}

	// else no, so move along...
      }

      // sequence image does NOT exist, so we can't use this sequence

      else {
	printf("NO %i: %s\n", seqnum, seqfullpathname.c_str());
 	leave_out_set_member(iter, iter_next, idx_set);
	left_out = true;
	break;
      }
      //      printf("%i %s\n", index - index_offset, logdir.c_str());
    }

    // go to next GT image

    if (!left_out)
      iter++;
  }

  printf("after: %i\n", (int) idx_set.size());
  fflush(stdout);
}

//----------------------------------------------------------------------------

char *UD_datetime_string()
{
  time_t ltime;
  struct tm *today;
  char *s;
  
  s = (char *) calloc(MAX_STRLEN, sizeof(char));

  time(&ltime);
  today = localtime(&ltime);

  strftime(s, MAX_STRLEN, "%b_%d_%Y_%a_%I_%M_%S_%p", today);

  return s;
} 

//----------------------------------------------------------------------------

void write_traintest_vanilla_grayscale(string dir, float training_fraction)
{
  int num_training, x_new, y_new, x_left, y_top;
  Mat im;
  //  vector < pair <string, vector <Point> > > TrainTest;
  vector < pair <string, VertVect > > TrainTest;
  set<int>::iterator iter;
  stringstream ss, short_ss;
  string date_str;
  string train_path, test_path;
  float scale_factor = (float) output_width / (float) output_crop_width; 

  filter_for_traintest();

  date_str = string(UD_datetime_string());

  ss << "mkdir " << dir << "_" << date_str;
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/train";
  train_path = ss.str();
  ss.str("");
  ss << "mkdir " << train_path; 
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/test";
  test_path = ss.str();
  ss.str("");
  ss << "mkdir " << test_path; 
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  num_training = (int) rint(training_fraction * (float) FilteredVert_idx_set.size());
  x_left = CANONICAL_IMAGE_WIDTH / 2 - output_crop_width / 2;
  y_top = output_crop_top_y;

  // copy names from Filtered_idx_set into vector 
  
  TrainTest.clear();
  for (iter = FilteredVert_idx_set.begin(); iter != FilteredVert_idx_set.end(); iter++) 
    TrainTest.push_back(make_pair(Fullpathname_vect[*iter], Vert[*iter]));
  //TrainTest.push_back(make_pair(Idx_signature_vect[*iter], Vert[*iter]));

  // shuffle 

  random_shuffle(TrainTest.begin(), TrainTest.end());

  // open train, test vert files

  string train_vert_filename;
  string test_vert_filename;
  string params_filename;
  FILE *train_vert_fp, *test_vert_fp, *params_fp;

  ss.str("");
  ss << dir << "_" << date_str << "/trainvert.txt";
  train_vert_filename = ss.str();
  train_vert_fp = fopen(train_vert_filename.c_str(), "w");

  ss.str("");
  ss << dir << "_" << date_str << "/testvert.txt";
  test_vert_filename = ss.str();
  test_vert_fp = fopen(test_vert_filename.c_str(), "w");

  printf("%s\n%s\n", train_vert_filename.c_str(), test_vert_filename.c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/params.txt";
  params_filename = ss.str();
  params_fp = fopen(params_filename.c_str(), "w");

  fprintf(params_fp, "label type = FAR x left, FAR x right (ordered)\n");

  fprintf(params_fp, "crop width = %i\n", output_crop_width);
  fprintf(params_fp, "crop height = %i\n", output_crop_height);
  fprintf(params_fp, "crop top y = %i\n", output_crop_top_y);

  fprintf(params_fp, "output width = %i\n", output_width);
  fprintf(params_fp, "output height = %i\n", output_height);

  fprintf(params_fp, "similarity threshold = %.3f\n", SIMILARITY_THRESHOLD);
  fprintf(params_fp, "training fraction = %.3f\n", training_fraction);

  fprintf(params_fp, "input data = %s\n", s_imagedirs.c_str());

  fprintf(params_fp, "# raw images = %i (current)\n", (int) Fullpathname_vect.size());
  fprintf(params_fp, "# bad images = %i (current)\n", (int) Bad_idx_set.size());
  fprintf(params_fp, "# GT images = %i (current)\n", (int) Vert_idx_set.size());
  fprintf(params_fp, "# filtered images = %i (current)\n", (int) FilteredVert_idx_set.size());

  fclose(params_fp);

  // iterate through and load each image into "im"

  for (int i = 0; i < TrainTest.size(); i++) {

    //    printf("new %i: reading %s\n", i, TrainTest[i].first.c_str());

    im = smart_imread(TrainTest[i].first);

    // for each, crop, resize, and WRITE
    
    Rect crop_rect(im.cols / 2 - output_crop_width / 2, output_crop_top_y, 
		   output_crop_width, output_crop_height);
    Mat crop_im = im(crop_rect);
    Mat output_cubic_im = Mat(output_height, output_width, CV_8UC3);
    resize(crop_im, output_cubic_im, output_cubic_im.size(), 0, 0, INTER_CUBIC);
    Mat output_im = Mat(output_height, output_width, CV_8UC1);
    
    cvtColor(output_cubic_im, output_im, cv::COLOR_RGB2GRAY);
  
    // WRITE NEW IMAGE, NEW VERTS WITH NEW NAME!
    
    string imname;
    FILE *fp;

    ss.str("");
    short_ss.str("");

    if (i < num_training) {
      //printf("TRAIN: ");
      ss << train_path << "/" << setw(6) << setfill('0') << i << ".jpg";
      short_ss << "train/" << setw(6) << setfill('0') << i << ".jpg";
      fp = train_vert_fp;
    }
    else {
      //printf("TEST: ");
      ss << test_path << "/" << setw(6) << setfill('0') << i - num_training << ".jpg";
      short_ss << "test/" << setw(6) << setfill('0') << i - num_training << ".jpg";
      fp = test_vert_fp;
    }

    imname = ss.str();

    //    printf("%s\n", imname.c_str());
    imwrite(imname.c_str(), output_im);

    //    fprintf(fp, "%s, %s, ", imname.c_str(), TrainTest[i].first.c_str());
    //    fprintf(fp, "%s, %s, ", short_ss.str().c_str(), TrainTest[i].first.c_str());

    string sig = fullpathname_to_signature(TrainTest[i].first);
    //printf("writing %s\n", sig.c_str()); fflush(stdout);
    fprintf(fp, "%s, %s, ", short_ss.str().c_str(), sig.c_str());
    fflush(fp);

    // calculate new vert coords and write them
    
    vector <int> Xval;

    for (int j = 0; j < TrainTest[i].second.size(); j++) {

      if (TrainTest[i].second[j].y == IMAGE_ROW_FAR) {

	x_new = (int) rint(scale_factor * (float) (TrainTest[i].second[j].x - x_left));
	y_new = (int) rint(scale_factor * (float) (TrainTest[i].second[j].y - y_top));

	//	printf("%i: %i, %i\n", j, x_new, y_new); fflush(stdout);
 
	// WRITE THEM

	//	if (!(j % 2)) {
	Xval.push_back(x_new);
	  //}
	//else
	// fprintf(fp, "%i\n", x_new);
      }
    }

    sort(Xval.begin(), Xval.end());

    //    line(output_cubic_im, Point(Xval[0], y_new), Point(Xval[1], y_new), Scalar(0, 255, 0), 2);

    fprintf(fp, "%i, %i\n", Xval[0], Xval[1]);
    fflush(fp);

    // wait to write image so we can draw on it for debugging

    // imwrite(imname.c_str(), output_cubic_im);
    
  }

  // close train/test vert files
  
  fclose(train_vert_fp);
  fclose(test_vert_fp);

}



//----------------------------------------------------------------------------

// output_color_im = crop_and_scale_image(im, output_crop_top_y, output_crop_width, output_crop_height, output_width, output_height);

Mat crop_and_scale_image(Mat & input_im, 
			 int crop_x_left, int crop_x_right,
			 int crop_top_y, int crop_height,
			 int out_width, int out_height)
{
  /*
  printf("crop and scale\n");

  printf("%i %i %i %i %i %i\n", 
	 crop_x_left,  crop_x_right,
	 crop_top_y,  crop_height,
	 out_width,  out_height);
  fflush(stdout);
  */

  Rect crop_rect(crop_x_left, crop_top_y, crop_x_right - crop_x_left, crop_height);
  //  Rect crop_rect(input_im.cols / 2 - crop_width / 2, crop_top_y, crop_width, crop_height);
  Mat crop_im = input_im(crop_rect);
  Mat out_im = Mat(out_height, out_width, CV_8UC3);
  resize(crop_im, out_im, out_im.size(), 0, 0, INTER_CUBIC);

  return out_im;
}

//----------------------------------------------------------------------------

// FAR ONLY

// need to make sure trail edges are still in image after distortion

// when we are actually using color in the network, add random hue/saturation changes

// input_im is raw image

// Xval is AFTER crop (subtract x_left) and scale (multiply by crop_scale_factor)
// x_left, x_right, x_left_new, and x_right_new are BEFORE crop

Mat far_distort_one_image(Mat & input_im, const vector <int> & Xval, vector <int> & Xval_new)
{
  int i;
  float crop_scale_factor = (float) output_width / (float) output_crop_width;
  int random_crop_shift_factor;

  int x_left = input_im.cols / 2 - output_crop_width / 2;
  int x_right = input_im.cols / 2 + output_crop_width / 2;

  //  printf("old x: %i, %i\n", x_left, x_right);
  //  fflush(stdout);

  //  printf("delta = %i\n", distort_max_horizontal_delta);

  // (1) shift and/or scale -- CAN CHANGE Xval
  // if random transform takes left OR right edge out of image, do it AGAIN!!

  bool no_good = true;
  int x_left_new, x_right_new;

  while (no_good) {

    x_left_new = ranged_uniform_int_UD_Random(x_left - distort_max_horizontal_delta, x_left + distort_max_horizontal_delta); 
    x_right_new = ranged_uniform_int_UD_Random(x_right - distort_max_horizontal_delta, x_right + distort_max_horizontal_delta); 
    
    random_crop_shift_factor = x_left_new - x_left;   
    
    float shift_factor = crop_scale_factor * (float) random_crop_shift_factor;
    float scale_factor = (float) (x_right - x_left) / (float) (x_right_new - x_left_new);    
    
    // transform each edge

    no_good = false;

    for (i = 0; i < Xval.size(); i++) {

      Xval_new[i] = (int) rint(((float) Xval[i] - shift_factor) * scale_factor);

      // ** check if new edge location is outside image **

      if (Xval_new[i] < 0 || Xval_new[i] >= output_width)
	no_good = true;
    }
  }

  Mat output_im = crop_and_scale_image(input_im, 
				       x_left_new, x_right_new, 
				       output_crop_top_y, output_crop_height, 
				       output_width, output_height);

  // (2) flip horizontally?   CAN CHANGE Xval

  int temp;

  if (probability_UD_Random(distort_horizontal_flip_prob)) {

    flip(output_im, output_im, 1);

    for (i = 0; i < Xval.size(); i++)
      Xval_new[i] = output_im.cols - Xval_new[i] - 1;

    // swap left and right -- aka sort again 

    temp = Xval_new[1];
    Xval_new[1] = Xval_new[0];
    Xval_new[0] = temp;

  }

  // (3) random contrast/brightness change -- does NOT change Xval

  float contrast_factor = ranged_uniform_UD_Random(1.0 - distort_max_contrast_delta, 1.0 + distort_max_contrast_delta);   // 1 is no change
  float brightness_factor = ranged_uniform_UD_Random(-distort_max_brightness_delta, distort_max_brightness_delta);        // 0 is no change

  output_im.convertTo(output_im, -1, contrast_factor, brightness_factor);

  // done

  return output_im;
}


//----------------------------------------------------------------------------

// FAR AND NEAR

// need to make sure trail edges are still in image after distortion

// when we are actually using color in the network, add random hue/saturation changes?

// input_im is raw image (after smart_imread())

// Xval is AFTER crop (subtract x_left) and scale (multiply by crop_scale_factor)
// x_left, x_right, x_left_new, and x_right_new are BEFORE crop

Mat far_near_distort_one_image(Mat & input_im, 
			       const vector <int> & FarXval, vector <int> & FarXval_new,
			       const vector <int> & NearXval, vector <int> & NearXval_new)
{
  int i;
  float crop_scale_factor = (float) output_width / (float) output_crop_width;
  int random_crop_shift_factor;

  int x_left = input_im.cols / 2 - output_crop_width / 2;
  int x_right = input_im.cols / 2 + output_crop_width / 2;

  //  printf("old x: %i, %i\n", x_left, x_right);
  //  fflush(stdout);

  //  printf("delta = %i\n", distort_max_horizontal_delta);

  // (1) shift and/or scale -- CAN CHANGE FarXval
  // if random transform takes left OR right edge out of image, do it AGAIN!!

  bool no_good = true;
  int x_left_new, x_right_new;

  while (no_good) {

    x_left_new = ranged_uniform_int_UD_Random(x_left - distort_max_horizontal_delta, x_left + distort_max_horizontal_delta); 
    x_right_new = ranged_uniform_int_UD_Random(x_right - distort_max_horizontal_delta, x_right + distort_max_horizontal_delta); 
    
    random_crop_shift_factor = x_left_new - x_left;   
    
    float shift_factor = crop_scale_factor * (float) random_crop_shift_factor;
    float scale_factor = (float) (x_right - x_left) / (float) (x_right_new - x_left_new);    
    
    // transform each edge

    no_good = false;

    for (i = 0; i < FarXval.size(); i++) {

      FarXval_new[i] = (int) rint(((float) FarXval[i] - shift_factor) * scale_factor);

      // ** check if new edge location is outside image **

      if (FarXval_new[i] < 0 || FarXval_new[i] >= output_width)
	no_good = true;
    }

    for (i = 0; i < NearXval.size(); i++) {

      NearXval_new[i] = (int) rint(((float) NearXval[i] - shift_factor) * scale_factor);

      // ** check if new edge location is outside image **

      if (NearXval_new[i] < 0 || NearXval_new[i] >= output_width)
	no_good = true;
    }
  }

  Mat output_im = crop_and_scale_image(input_im, 
				       x_left_new, x_right_new, 
				       output_crop_top_y, output_crop_height, 
				       output_width, output_height);

  // (2) flip horizontally?   CAN CHANGE FarXval

  int temp;

  if (probability_UD_Random(distort_horizontal_flip_prob)) {

    flip(output_im, output_im, 1);

    for (i = 0; i < FarXval.size(); i++)
      FarXval_new[i] = output_im.cols - FarXval_new[i] - 1;

    for (i = 0; i < NearXval.size(); i++)
      NearXval_new[i] = output_im.cols - NearXval_new[i] - 1;

    // swap left and right -- aka sort again 

    temp = FarXval_new[1];
    FarXval_new[1] = FarXval_new[0];
    FarXval_new[0] = temp;

    temp = NearXval_new[1];
    NearXval_new[1] = NearXval_new[0];
    NearXval_new[0] = temp;

  }

  // (3) random contrast/brightness change -- does NOT change FarXval

  float contrast_factor = ranged_uniform_UD_Random(1.0 - distort_max_contrast_delta, 1.0 + distort_max_contrast_delta);   // 1 is no change
  float brightness_factor = ranged_uniform_UD_Random(-distort_max_brightness_delta, distort_max_brightness_delta);        // 0 is no change

  output_im.convertTo(output_im, -1, contrast_factor, brightness_factor);

  // done

  return output_im;
}

//----------------------------------------------------------------------------

// NON-TRAIL (BAD)

// when we are actually using color in the network, add random hue/saturation changes

// input_im is raw image

Mat nontrail_distort_one_image(Mat & input_im)
{
  int i;
  float crop_scale_factor = (float) output_width / (float) output_crop_width;
  int random_crop_shift_factor;

  int x_left = input_im.cols / 2 - output_crop_width / 2;
  int x_right = input_im.cols / 2 + output_crop_width / 2;

  // (1) shift and/or scale -- CAN CHANGE FarXval
  // if random transform takes left OR right edge out of image, do it AGAIN!!

  int x_left_new, x_right_new;

  x_left_new = ranged_uniform_int_UD_Random(x_left - distort_max_horizontal_delta, x_left + distort_max_horizontal_delta); 
  x_right_new = ranged_uniform_int_UD_Random(x_right - distort_max_horizontal_delta, x_right + distort_max_horizontal_delta); 
    
  random_crop_shift_factor = x_left_new - x_left;   
    
  float shift_factor = crop_scale_factor * (float) random_crop_shift_factor;
  float scale_factor = (float) (x_right - x_left) / (float) (x_right_new - x_left_new);    
  
  Mat output_im = crop_and_scale_image(input_im, 
				       x_left_new, x_right_new, 
				       output_crop_top_y, output_crop_height, 
				       output_width, output_height);

  // (2) flip horizontally?   CAN CHANGE FarXval

  if (probability_UD_Random(distort_horizontal_flip_prob)) 
    flip(output_im, output_im, 1);

  // (3) random contrast/brightness change -- does NOT change FarXval

  float contrast_factor = ranged_uniform_UD_Random(1.0 - distort_max_contrast_delta, 1.0 + distort_max_contrast_delta);   // 1 is no change
  float brightness_factor = ranged_uniform_UD_Random(-distort_max_brightness_delta, distort_max_brightness_delta);        // 0 is no change

  output_im.convertTo(output_im, -1, contrast_factor, brightness_factor);

  // done

  return output_im;
}

//----------------------------------------------------------------------------

// FAR ONLY
// generate all distortions for a single input image

void generate_far_training_image_distortions(Mat & input_im, string & input_im_signature, vector <int> & Xval,
					 int num_distortions,
					 int & current_index, 
					 string & current_path,
					 const vector <int> & Idx_distorted,
					 vector <string> & Line_distorted,
					 FILE *distort_fp = NULL)
{
  int i;
  Mat output_im;
  stringstream ss, short_ss, line_ss;
  vector <int> Xval_new;

  Xval_new.resize(Xval.size());

  for (i = 0; i < num_distortions; i++) {

    // generate ONE randomly-distorted training image from input_im

    output_im = far_distort_one_image(input_im, Xval, Xval_new);

    // if trail edges are still in image, write info to distorted_train_vert_fp

    // write image to distorted_train directory

    ss.str("");
    short_ss.str("");
    line_ss.str("");

    ss << current_path << "/" << setw(6) << setfill('0') << Idx_distorted[current_index] << ".jpg";
    short_ss << "distorted_train/" << setw(6) << setfill('0') << Idx_distorted[current_index] << ".jpg";

    //    printf("%s\n", ss.str().c_str());
    //    printf("%s\n", short_ss.str().c_str());

    //    printf("%s\n", imname.c_str());

    if (distort_fp) {

      //      fprintf(distort_fp, "%s, %s, ", short_ss.str().c_str(), input_im_signature.c_str());

      // write new vert coords 

      //      line(output_im, Point(Xval_new[0], output_im.rows/2), Point(Xval_new[1], output_im.rows/2), Scalar(0, 255, 0), 2);

      //      fprintf(distort_fp, "%i, %i\n", Xval_new[0], Xval_new[1]);
      //      fflush(distort_fp);


    }

    line_ss << short_ss.str().c_str() << ", " << input_im_signature.c_str() << ", " << Xval_new[0] << ", " << Xval_new[1];
    Line_distorted[Idx_distorted[current_index]] = line_ss.str();

    // delay writing so we can draw debug overlay 

    imwrite(ss.str().c_str(), output_im);

    // --> current directory + "/distorted_train"
    
    current_index++;
  }

}

//----------------------------------------------------------------------------

// order: left, right, flip, contrast, brightness

void print_distortion(FILE *fp, DistortParams & D)
{
  fprintf(fp, "%i, %i, %i, %.3f, %.3f\n", D.x_left_new, D.x_right_new, D.do_horizontal_flip, D.contrast_factor, D.brightness_factor);
  fflush(fp);
}

//----------------------------------------------------------------------------

// FAR AND NEAR

// compute and save distortion factors, but don't actually apply them (yet)
// checks to make sure trail edges are still in image after distortion -- may not be true for rest of sequence, of course

// input_im is raw image

// Xval is AFTER crop (subtract x_left) and scale (multiply by crop_scale_factor)
// x_left, x_right, x_left_new, and x_right_new are BEFORE crop

DistortParams precompute_far_near_distort_one_image(Mat & input_im, const vector <int> & FarXval, const vector <int> & NearXval)
{
  DistortParams D;

  vector <int> FarXval_new, NearXval_new;

  FarXval_new.resize(FarXval.size());
  NearXval_new.resize(NearXval.size());

  int i;
  float crop_scale_factor = (float) output_width / (float) output_crop_width;
  int random_crop_shift_factor;

  int x_left = input_im.cols / 2 - output_crop_width / 2;
  int x_right = input_im.cols / 2 + output_crop_width / 2;

  // (1) shift and/or scale -- CAN CHANGE FarXval
  // if random transform takes left OR right edge out of image, do it AGAIN!!

  bool no_good = true;

  //  printf("original %i %i\n", x_left, x_right); fflush(stdout);

  while (no_good) {

    D.x_left_new = ranged_uniform_int_UD_Random(x_left - distort_max_horizontal_delta, x_left + distort_max_horizontal_delta); 
    D.x_right_new = ranged_uniform_int_UD_Random(x_right - distort_max_horizontal_delta, x_right + distort_max_horizontal_delta); 

    //    printf("new %i %i\n", D.x_left_new, D.x_right_new); fflush(stdout);
    
    random_crop_shift_factor = D.x_left_new - x_left;   
    
    float shift_factor = crop_scale_factor * (float) random_crop_shift_factor;
    float scale_factor = (float) (x_right - x_left) / (float) (D.x_right_new - D.x_left_new);    
    
    // transform each edge

    no_good = false;

    for (i = 0; i < FarXval.size(); i++) {

      FarXval_new[i] = (int) rint(((float) FarXval[i] - shift_factor) * scale_factor);

      // ** check if new edge location is outside image **

      if (FarXval_new[i] < 0 || FarXval_new[i] >= output_width)
	no_good = true;
    }

    for (i = 0; i < NearXval.size(); i++) {

      NearXval_new[i] = (int) rint(((float) NearXval[i] - shift_factor) * scale_factor);

      // ** check if new edge location is outside image **

      if (NearXval_new[i] < 0 || NearXval_new[i] >= output_width)
	no_good = true;
    }
  }

  // (2) flip horizontally?   CAN CHANGE FarXval

  D.do_horizontal_flip = probability_UD_Random(distort_horizontal_flip_prob);

  // (3) random contrast/brightness change -- does NOT change FarXval

  D.contrast_factor = ranged_uniform_UD_Random(1.0 - distort_max_contrast_delta, 1.0 + distort_max_contrast_delta);   // 1 is no change
  D.brightness_factor = ranged_uniform_UD_Random(-distort_max_brightness_delta, distort_max_brightness_delta);        // 0 is no change

  return D;
}

//----------------------------------------------------------------------------

// this function is expecting FarXval and NearXval to be in cropped/scaled image coordinates 

// input_im should be the dimensions/orientation returned by smart_imread()

Mat apply_precomputed_distortion(DistortParams & D,
				 Mat & input_im, 
				 const vector <int> & FarXval, vector <int> & FarXval_new,
				 const vector <int> & NearXval, vector <int> & NearXval_new)
{
  int i;
  float crop_scale_factor = (float) output_width / (float) output_crop_width;

  int x_left = input_im.cols / 2 - output_crop_width / 2;
  int x_right = input_im.cols / 2 + output_crop_width / 2;

  //  printf("csf = %.3f\n", crop_scale_factor);
  //  printf("x left = %i, right = %i\n", x_left, x_right);

  //  print_distortion(stdout, D);

  // (1) shift and/or scale -- CAN CHANGE FarXval
  // if random transform takes left OR right edge out of image, do it AGAIN!!
    
  float shift_factor = crop_scale_factor * (float) (D.x_left_new - x_left);
  float scale_factor = (float) (x_right - x_left) / (float) (D.x_right_new - D.x_left_new);    

  //  printf("shift = %.3f, scale = %.3f\n", shift_factor, scale_factor);
  
  // transform each edge
  
  for (i = 0; i < FarXval.size(); i++) 
    FarXval_new[i] = (int) rint(((float) FarXval[i] - shift_factor) * scale_factor);
  
  for (i = 0; i < NearXval.size(); i++) 
    NearXval_new[i] = (int) rint(((float) NearXval[i] - shift_factor) * scale_factor);
  
  Mat output_im = crop_and_scale_image(input_im, 
				       D.x_left_new, D.x_right_new, 
				       output_crop_top_y, output_crop_height, 
				       output_width, output_height);

  //  printf("output w, h = %i x %i\n", output_im.cols, output_im.rows);

  // (2) flip horizontally?   CAN CHANGE FarXval

  int temp;

  if (D.do_horizontal_flip) {

    flip(output_im, output_im, 1);

    for (i = 0; i < FarXval.size(); i++)
      FarXval_new[i] = output_im.cols - FarXval_new[i] - 1;

    for (i = 0; i < NearXval.size(); i++)
      NearXval_new[i] = output_im.cols - NearXval_new[i] - 1;

    // swap left and right -- aka sort again 

    temp = FarXval_new[1];
    FarXval_new[1] = FarXval_new[0];
    FarXval_new[0] = temp;

    temp = NearXval_new[1];
    NearXval_new[1] = NearXval_new[0];
    NearXval_new[0] = temp;

  }

  // (3) random contrast/brightness change -- does NOT change FarXval

  output_im.convertTo(output_im, -1, D.contrast_factor, D.brightness_factor);

  // done

  return output_im;
}

//----------------------------------------------------------------------------

// generate set of random geometric and photometric distortions from/for ONE image.
// makes sure that none of them cause the trail vertices to leave the image

void precompute_far_near_training_image_distortions(Mat & input_im, vector <int> & FarXval, vector <int> & NearXval, int num_distortions,
						    vector <DistortParams> & V_distort)
{
  V_distort.clear();

  for (int i = 0; i < num_distortions; i++) 
    V_distort.push_back(precompute_far_near_distort_one_image(input_im, FarXval, NearXval));

}

//----------------------------------------------------------------------------

// FAR AND NEAR
// generate all distortions for a single input image

void generate_far_near_training_image_distortions(Mat & input_im, string & input_im_signature, 
						  vector <int> & FarXval, vector <int> & NearXval,
						  int num_distortions,
						  int & current_index, 
						  string & current_path,
						  const vector <int> & Idx_distorted,
						  vector <string> & Line_distorted,
						  FILE *distort_fp = NULL)
{
  int i;
  Mat output_im;
  stringstream ss, short_ss, line_ss;
  vector <int> FarXval_new, NearXval_new;

  float y_scale_factor = (float) output_height / (float) output_crop_height; 

  int y_far_new =  (int) rint(y_scale_factor * (float) (IMAGE_ROW_FAR - output_crop_top_y));
  int y_near_new =  (int) rint(y_scale_factor * (float) (IMAGE_ROW_NEAR - output_crop_top_y));
	
  printf("far %i, near %i\n", y_far_new, y_near_new);
  //  fflush(stdout);
  //  exit(1);

  FarXval_new.resize(FarXval.size());
  NearXval_new.resize(NearXval.size());

  for (i = 0; i < num_distortions; i++) {

    // generate ONE randomly-distorted training image from input_im

    output_im = far_near_distort_one_image(input_im, 
					   FarXval, FarXval_new,
					   NearXval, NearXval_new);

    // if trail edges are still in image, write info to distorted_train_vert_fp

    // write image to distorted_train directory

    ss.str("");
    short_ss.str("");
    line_ss.str("");

    ss << current_path << "/" << setw(6) << setfill('0') << Idx_distorted[current_index] << ".jpg";
    short_ss << "distorted_train/" << setw(6) << setfill('0') << Idx_distorted[current_index] << ".jpg";

    //    printf("%s\n", ss.str().c_str());
    //    printf("%s\n", short_ss.str().c_str());

    //    printf("%s\n", imname.c_str());

    if (distort_fp) {

      //      fprintf(distort_fp, "%s, %s, ", short_ss.str().c_str(), input_im_signature.c_str());

      // write new vert coords 

      // overlay line segments for debugging
      //      line(output_im, Point(FarXval_new[0], y_far_new), Point(FarXval_new[1], y_far_new), Scalar(0, 255, 0), 2);
      //      line(output_im, Point(NearXval_new[0], y_near_new), Point(NearXval_new[1], y_near_new), Scalar(0, 255, 0), 2);

      //      fprintf(distort_fp, "%i, %i\n", FarXval_new[0], FarXval_new[1]);
      //      fflush(distort_fp);


    }

    line_ss << short_ss.str().c_str() << ", " << input_im_signature.c_str() << ", " << FarXval_new[0] << ", " << FarXval_new[1] << ", " << NearXval_new[0] << ", " << NearXval_new[1];
    Line_distorted[Idx_distorted[current_index]] = line_ss.str();

    // delay writing so we can draw debug overlay 

    imwrite(ss.str().c_str(), output_im);

    // --> current directory + "/distorted_train"
    
    current_index++;
  }

}

//----------------------------------------------------------------------------

// NON-TRAIL (BAD)
// generate all distortions for a single input image

void generate_nontrail_training_image_distortions(Mat & input_im, string & input_im_signature, 
						  int num_distortions,
						  int & current_index, 
						  string & current_path,
						  const vector <int> & Idx_distorted,
						  vector <string> & Line_distorted,
						  FILE *distort_fp = NULL)
{
  int i;
  Mat output_im;
  stringstream ss, short_ss, line_ss;

  float y_scale_factor = (float) output_height / (float) output_crop_height; 

  for (i = 0; i < num_distortions; i++) {

    // generate ONE randomly-distorted training image from input_im

    output_im = nontrail_distort_one_image(input_im);

    // write image to distorted_train directory

    ss.str("");
    short_ss.str("");
    line_ss.str("");

    ss << current_path << "/" << setw(6) << setfill('0') << Idx_distorted[current_index] << ".jpg";
    short_ss << "nontrail_distorted_train/" << setw(6) << setfill('0') << Idx_distorted[current_index] << ".jpg";

    //    printf("%s\n", ss.str().c_str());
    //    printf("%s\n", short_ss.str().c_str());

    //    printf("%s\n", imname.c_str());

    if (distort_fp) {

      //      fprintf(distort_fp, "%s, %s, ", short_ss.str().c_str(), input_im_signature.c_str());

      // write new vert coords 

      // overlay line segments for debugging
      //      line(output_im, Point(FarXval_new[0], y_far_new), Point(FarXval_new[1], y_far_new), Scalar(0, 255, 0), 2);
      //      line(output_im, Point(NearXval_new[0], y_near_new), Point(NearXval_new[1], y_near_new), Scalar(0, 255, 0), 2);

      //      fprintf(distort_fp, "%i, %i\n", FarXval_new[0], FarXval_new[1]);
      //      fflush(distort_fp);


    }

    line_ss << short_ss.str().c_str() << ", " << input_im_signature.c_str();
    Line_distorted[Idx_distorted[current_index]] = line_ss.str();

    // delay writing so we can draw debug overlay 

    imwrite(ss.str().c_str(), output_im);

    // --> current directory + "/distorted_train"
    
    current_index++;
  }

}

//----------------------------------------------------------------------------

// top only
// with distortions

void write_traintest_chocolate_color(string dir, float training_fraction)
{
  int i, distort_index, num_training, x_new, y_new, x_left, y_top;
  Mat im;
  //  vector < pair <string, vector <Point> > > TrainTest;
  vector < pair <string, VertVect > > TrainTest;
  set<int>::iterator iter;
  stringstream ss, short_ss;
  string date_str;
  string train_path, distorted_train_path, test_path;
  float scale_factor = (float) output_width / (float) output_crop_width;    // e.g., 160 / 250

  filter_for_traintest();

  date_str = string(UD_datetime_string());

  ss << "mkdir " << dir << "_" << date_str;
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/train";
  train_path = ss.str();
  ss.str("");
  ss << "mkdir " << train_path; 
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/distorted_train";
  distorted_train_path = ss.str();
  ss.str("");
  ss << "mkdir " << distorted_train_path; 
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/test";
  test_path = ss.str();
  ss.str("");
  ss << "mkdir " << test_path; 
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  num_training = (int) rint(training_fraction * (float) FilteredVert_idx_set.size());
  x_left = CANONICAL_IMAGE_WIDTH / 2 - output_crop_width / 2;
  y_top = output_crop_top_y;

  // copy names from Filtered_idx_set into vector 
  
  TrainTest.clear();
  for (iter = FilteredVert_idx_set.begin(); iter != FilteredVert_idx_set.end(); iter++) 
    TrainTest.push_back(make_pair(Fullpathname_vect[*iter], Vert[*iter]));

  // shuffle 

  random_shuffle(TrainTest.begin(), TrainTest.end());

  // open train, test vert files

  string train_vert_filename;
  string distorted_train_vert_filename;
  string test_vert_filename;
  string params_filename;
  FILE *train_vert_fp, *distorted_train_vert_fp, *test_vert_fp, *params_fp;

  ss.str("");
  ss << dir << "_" << date_str << "/trainvert.txt";
  train_vert_filename = ss.str();
  train_vert_fp = fopen(train_vert_filename.c_str(), "w");

  ss.str("");
  ss << dir << "_" << date_str << "/distorted_trainvert.txt";
  distorted_train_vert_filename = ss.str();
  distorted_train_vert_fp = fopen(distorted_train_vert_filename.c_str(), "w");

  ss.str("");
  ss << dir << "_" << date_str << "/testvert.txt";
  test_vert_filename = ss.str();
  test_vert_fp = fopen(test_vert_filename.c_str(), "w");

  printf("%s\n%s\n", train_vert_filename.c_str(), test_vert_filename.c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/params.txt";
  params_filename = ss.str();
  params_fp = fopen(params_filename.c_str(), "w");

  fprintf(params_fp, "label type = FAR x left, FAR x right (ordered)\n");

  fprintf(params_fp, "crop width = %i\n", output_crop_width);
  fprintf(params_fp, "crop height = %i\n", output_crop_height);
  fprintf(params_fp, "crop top y = %i\n", output_crop_top_y);

  fprintf(params_fp, "output width = %i\n", output_width);
  fprintf(params_fp, "output height = %i\n", output_height);

  fprintf(params_fp, "similarity threshold = %.3f\n", SIMILARITY_THRESHOLD);
  fprintf(params_fp, "training fraction = %.3f\n", training_fraction);

  fprintf(params_fp, "input data = %s\n", s_imagedirs.c_str());

  fprintf(params_fp, "# raw images = %i (current)\n", (int) Fullpathname_vect.size());
  fprintf(params_fp, "# bad images = %i (current)\n", (int) Bad_idx_set.size());
  fprintf(params_fp, "# GT images = %i (current)\n", (int) Vert_idx_set.size());
  fprintf(params_fp, "# filtered images = %i (current)\n", (int) FilteredVert_idx_set.size());

  fclose(params_fp);

  // generate and shuffle indices of distorted training images
  
  vector <string> Line_distorted;
  vector <int> Idx_distorted;
  
  Line_distorted.resize(num_training * distort_num_per_image);
  Idx_distorted.resize(num_training * distort_num_per_image);
  for (int z = 0; z < Idx_distorted.size(); z++)
    Idx_distorted[z] = z;
  random_shuffle(Idx_distorted.begin(), Idx_distorted.end());

  // iterate through and load each image into "im"

  for (i = 0, distort_index = 0; i < TrainTest.size(); i++) {

    //    printf("new %i: reading %s\n", i, TrainTest[i].first.c_str());

    im = smart_imread(TrainTest[i].first);
    string imsig = fullpathname_to_signature(TrainTest[i].first);

    // for each, crop, resize, and WRITE
    
    Mat output_color_im = crop_and_scale_image(im, 
					       im.cols / 2 - output_crop_width / 2, im.cols / 2 + output_crop_width / 2,
					       output_crop_top_y, output_crop_height, 
					       output_width, output_height);
    //    Mat output_color_im = crop_and_scale_image(im, output_crop_top_y, output_crop_width, output_crop_height, output_width, output_height);

    /*
    Rect crop_rect(im.cols / 2 - output_crop_width / 2, output_crop_top_y, 
		   output_crop_width, output_crop_height);
    Mat crop_im = im(crop_rect);
    Mat output_color_im = Mat(output_height, output_width, CV_8UC3);
    resize(crop_im, output_color_im, output_color_im.size(), 0, 0, INTER_CUBIC);
    //    Mat output_im = Mat(output_height, output_width, CV_8UC1);
    */

    //    cvtColor(output_cubic_im, output_im, cv::COLOR_RGB2GRAY);
  
    // WRITE NEW IMAGE, NEW VERTS WITH NEW NAME!
    
    string imname;
    FILE *fp;
    vector <int> Xval;

    ss.str("");
    short_ss.str("");

    // calculate new vert coords 
    
    for (int j = 0; j < TrainTest[i].second.size(); j++) {

      if (TrainTest[i].second[j].y == IMAGE_ROW_FAR) {

	x_new = (int) rint(scale_factor * (float) (TrainTest[i].second[j].x - x_left));
	y_new = (int) rint(scale_factor * (float) (TrainTest[i].second[j].y - y_top));

	//	printf("%i: %i, %i\n", j, x_new, y_new); fflush(stdout);
 
	// WRITE THEM

	//	if (!(j % 2)) {
	Xval.push_back(x_new);
	  //}
	//else
	// fprintf(fp, "%i\n", x_new);
      }
    }

    sort(Xval.begin(), Xval.end());

    // path info

    if (i < num_training) {

      generate_far_training_image_distortions(im, imsig, Xval,
					      distort_num_per_image, distort_index, distorted_train_path,
					      Idx_distorted,
					      Line_distorted,
					      distorted_train_vert_fp);

      //printf("TRAIN: ");
      ss << train_path << "/" << setw(6) << setfill('0') << i << ".jpg";
      short_ss << "train/" << setw(6) << setfill('0') << i << ".jpg";
      fp = train_vert_fp;
    }
    else {
      //printf("TEST: ");
      ss << test_path << "/" << setw(6) << setfill('0') << i - num_training << ".jpg";
      short_ss << "test/" << setw(6) << setfill('0') << i - num_training << ".jpg";
      fp = test_vert_fp;
    }

    imname = ss.str();

    //    printf("%s\n", imname.c_str());
    imwrite(imname.c_str(), output_color_im);

    //    fprintf(fp, "%s, %s, ", imname.c_str(), TrainTest[i].first.c_str());
    //    fprintf(fp, "%s, %s, ", short_ss.str().c_str(), TrainTest[i].first.c_str());

    //printf("writing %s\n", imsig.c_str()); fflush(stdout);
    fprintf(fp, "%s, %s, ", short_ss.str().c_str(), imsig.c_str());
    fflush(fp);

    // write new vert coords 

    //    line(output_color_im, Point(Xval[0], y_new), Point(Xval[1], y_new), Scalar(0, 255, 0), 2);

    fprintf(fp, "%i, %i\n", Xval[0], Xval[1]);
    fflush(fp);

    // wait to write image so we can draw on it for debugging

    // imwrite(imname.c_str(), output_cubic_im);
    
  }

  for (i = 0; i < Line_distorted.size(); i++) {
    fprintf(distorted_train_vert_fp, "%s\n", Line_distorted[i].c_str());
    fflush(distorted_train_vert_fp);
  }

  // close train/test vert files
  
  fclose(train_vert_fp);
  fclose(test_vert_fp);
  fclose(distorted_train_vert_fp);
}


//----------------------------------------------------------------------------

// after applying shift/scale, how do vertices in V change?  put results in FarXval, NearXval

void calculate_new_vert_coords(VertVect & V, float scale_factor, int x_left, int y_top,
			       vector <int> & FarXval, vector <int> & NearXval)
{
  int j, x_new, y_new;
  
  for (int j = 0; j < V.size(); j++) {
    
    if (V[j].y == IMAGE_ROW_FAR) {
      
      x_new = (int) rint(scale_factor * (float) (V[j].x - x_left));
      y_new = (int) rint(scale_factor * (float) (V[j].y - y_top));
      
      // WRITE THEM
      
      FarXval.push_back(x_new);
    }
    
    else if (V[j].y == IMAGE_ROW_NEAR) {
      
      x_new = (int) rint(scale_factor * (float) (V[j].x - x_left));
      y_new = (int) rint(scale_factor * (float) (V[j].y - y_top));
      
      NearXval.push_back(x_new);
    }
  }
  
  sort(FarXval.begin(), FarXval.end());
  sort(NearXval.begin(), NearXval.end());
}

//----------------------------------------------------------------------------

// SEQUENCE
// top and bottom
// different crop
// with distortions

void write_traintest_peach_color(string dir, float training_fraction)
{
  int i, seqnum, distort_index, num_training, x_left, y_top;
  Mat im, seqim;
  vector < pair <string, VertVect > > TrainTest;
  set<int>::iterator iter;
  stringstream ss, short_ss;
  string date_str;
  string train_path, distorted_train_path, test_path;
  float scale_factor = (float) output_width / (float) output_crop_width;    // e.g., 160 / 250

  filter_for_traintest();   // starts with Vert_idx_set, outputs FilteredVert_idx_set
  filter_for_intact_sequence(FilteredVert_idx_set);

  //  filter_for_intact_sequence(Vert_idx_set);

  //  exit(1);

  date_str = string(UD_datetime_string());

  ss << "mkdir " << dir << "_" << date_str;
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/train";
  train_path = ss.str();
  ss.str("");
  ss << "mkdir " << train_path; 
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/test";
  test_path = ss.str();
  ss.str("");
  ss << "mkdir " << test_path; 
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/distorted_train";
  distorted_train_path = ss.str();
  ss.str("");
  ss << "mkdir " << distorted_train_path; 
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  // in case we didn't actually create FilteredVert for some reason, just copy it from Vert

  if (FilteredVert_idx_set.size() == 0) {
    FilteredVert_idx_set = Vert_idx_set;
  }

  // set some critical values

  x_left = CANONICAL_IMAGE_WIDTH / 2 - output_crop_width / 2;
  y_top = output_crop_top_y;
  num_training = (int) rint(training_fraction * (float) FilteredVert_idx_set.size());

  // copy names from Filtered_idx_set into vector 

  TrainTest.clear();
  for (iter = FilteredVert_idx_set.begin(); iter != FilteredVert_idx_set.end(); iter++) 
    TrainTest.push_back(make_pair(Fullpathname_vect[*iter], Vert[*iter]));

  // shuffle 

  random_shuffle(TrainTest.begin(), TrainTest.end());

  // open train, test vert files

  string train_vert_filename;
  string distorted_train_vert_filename;
  string test_vert_filename;
  string params_filename;
  FILE *train_vert_fp, *distorted_train_vert_fp, *test_vert_fp, *params_fp;

  ss.str("");
  ss << dir << "_" << date_str << "/trainvert.txt";
  train_vert_filename = ss.str();
  train_vert_fp = fopen(train_vert_filename.c_str(), "w");

  ss.str("");
  ss << dir << "_" << date_str << "/testvert.txt";
  test_vert_filename = ss.str();
  test_vert_fp = fopen(test_vert_filename.c_str(), "w");

  ss.str("");
  ss << dir << "_" << date_str << "/distorted_trainvert.txt";
  distorted_train_vert_filename = ss.str();
  distorted_train_vert_fp = fopen(distorted_train_vert_filename.c_str(), "w");

  printf("%s\n%s\n", train_vert_filename.c_str(), test_vert_filename.c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/params.txt";
  params_filename = ss.str();
  params_fp = fopen(params_filename.c_str(), "w");

  fprintf(params_fp, "label type = FAR x left, FAR x right, NEAR x left, NEAR x right (ordered -- SEQUENCE)\n");

  fprintf(params_fp, "crop width = %i\n", output_crop_width);
  fprintf(params_fp, "crop height = %i\n", output_crop_height);
  fprintf(params_fp, "crop top y = %i\n", output_crop_top_y);

  fprintf(params_fp, "output width = %i\n", output_width);
  fprintf(params_fp, "output height = %i\n", output_height);

  fprintf(params_fp, "similarity threshold = %.3f\n", SIMILARITY_THRESHOLD);
  fprintf(params_fp, "training fraction = %.3f\n", training_fraction);

  fprintf(params_fp, "sequence length = %i\n", PEACH_SEQUENCE_LENGTH);

  fprintf(params_fp, "input data = %s\n", s_imagedirs.c_str());

  fprintf(params_fp, "# raw images = %i (current)\n", (int) Fullpathname_vect.size());
  fprintf(params_fp, "# bad images = %i (current)\n", (int) Bad_idx_set.size());
  fprintf(params_fp, "# GT images = %i (current)\n", (int) Vert_idx_set.size());
  fprintf(params_fp, "# filtered images = %i (current)\n", (int) FilteredVert_idx_set.size());

  fclose(params_fp);

  // generate and shuffle indices of distorted training images
  
  vector <string> Line_distorted;
  vector <int> Idx_distorted;
  
  printf("num_training %i, distort_num_per_image %i\n", num_training, distort_num_per_image);
  fflush(stdout);

  Line_distorted.resize(num_training * distort_num_per_image);
  Idx_distorted.resize(num_training * distort_num_per_image);

  //  printf("Line_distorted size %i\n", Line_distorted.size());
  //  printf("Idx_distorted size %i\n", Idx_distorted.size());
  //  fflush(stdout);

  for (int z = 0; z < Idx_distorted.size(); z++)
    Idx_distorted[z] = z;
  random_shuffle(Idx_distorted.begin(), Idx_distorted.end());

  printf("traintest size %i\n", (int) TrainTest.size());
  fflush(stdout);

  // iterate through and load each image into "im"

  for (i = 0, distort_index = 0; i < TrainTest.size(); i++) {

    //    printf("new %i: reading %s\n", i, TrainTest[i].first.c_str());
    //   fflush(stdout);

    im = smart_imread(TrainTest[i].first);
    string imsig = fullpathname_to_signature(TrainTest[i].first);

    //    printf("%i: %s %i x %i\n", i, imsig.c_str(), im.cols, im.rows);
    //    fflush(stdout);

    vector <int> FarXval, NearXval;
    vector <int> FarXval_new, NearXval_new;


    calculate_new_vert_coords(TrainTest[i].second, scale_factor, x_left, y_top,
			      FarXval, NearXval);

    FarXval_new.resize(FarXval.size());
    NearXval_new.resize(NearXval.size());

    string imname;
    FILE *fp;

    // UNDISTORTED path info
    
    ss.str("");
    short_ss.str("");
    
    // this is a TRAINING image

    if (i < num_training) {
      ss << train_path << "/" << setw(6) << setfill('0') << i; //  << ".jpg";
      short_ss << "train/" << setw(6) << setfill('0') << i; //  << ".jpg";
      fp = train_vert_fp;
    }

    // this is a TESTING image

    else {
      ss << test_path << "/" << setw(6) << setfill('0') << i - num_training; // << ".jpg";
      short_ss << "test/" << setw(6) << setfill('0') << i - num_training; // << ".jpg";
      fp = test_vert_fp;
    }
    
    //      imname = ss.str();
    //    printf("ss = %s, short_ss = %s\n", ss.str().c_str(), short_ss.str().c_str());

    stringstream dir_ss, image_ss, line_ss, distort_ss;

    dir_ss.str("");
    dir_ss << "mkdir " << ss.str();

    // make directory in train/test for this sequence

    system(dir_ss.str().c_str());

    // write entry in test/train vert file, save image

    imname = ss.str();
      
    printf("imname %s\n", imname.c_str());

    //    imwrite(imname.c_str(), output_color_im);
      
    fprintf(fp, "%s, %s, ", short_ss.str().c_str(), imsig.c_str()); fflush(fp);
      
    //    line(output_color_im, Point(FarXval[0], y_new), Point(FarXval[1], y_new), Scalar(0, 255, 0), 2);
      
    fprintf(fp, "%i, %i, %i, %i\n", FarXval[0], FarXval[1], NearXval[0], NearXval[1]); fflush(fp);
      
    // write train/test image sequence here?

    int index = seqnum_from_signature(imsig);
    string logdir = logdir_from_signature(imsig);

    // this works...it just takes a while to run

    for (int index_offset = 0; index_offset <= PEACH_SEQUENCE_LENGTH; index_offset++) {

      int seqnum = index - index_offset;
      string seqfullpathname = fullpathname_from_logdir_and_seqnum(logdir, seqnum);

      seqim = smart_imread(seqfullpathname);

      // crop, resize, and WRITE
      
      Mat output_color_im = crop_and_scale_image(seqim, 
						 seqim.cols / 2 - output_crop_width / 2, seqim.cols / 2 + output_crop_width / 2,
						 output_crop_top_y, output_crop_height,
						 output_width, output_height);

      image_ss.str("");
      image_ss << ss.str() << "/" << setw(3) << setfill('0') << index_offset << ".jpg";

      // write image_ss

      imwrite(image_ss.str().c_str(), output_color_im);
    }

    // DISTORTIONS -- only do for training images

    if (i < num_training) {

      vector <DistortParams> V_distort;
      precompute_far_near_training_image_distortions(im, FarXval, NearXval, distort_num_per_image, V_distort);

      for (int d = 0; d < V_distort.size(); d++) {
	
	printf("distort index %i\n", distort_index);
	print_distortion(stdout, V_distort[d]);
	
	ss.str("");
	short_ss.str("");
	line_ss.str("");
	distort_ss.str("");

	ss << distorted_train_path << "/" << setw(6) << setfill('0') << Idx_distorted[distort_index]; // << ".jpg";
	short_ss << "distorted_train/" << setw(6) << setfill('0') << Idx_distorted[distort_index]; // << ".jpg";

	dir_ss.str("");
	dir_ss << "mkdir " << ss.str();

	// make directory in distorted_train for this sequence

	system(dir_ss.str().c_str());

	//	printf("%s\n", dir_ss.str().c_str());
	
	//	printf("ss %s, short_ss %s\n", ss.str().c_str(), short_ss.str().c_str());
							
	distort_ss << ss.str() << "/distort.txt";
	FILE *distort_fp = fopen(distort_ss.str().c_str(), "w");
	print_distortion(distort_fp, V_distort[d]);
	fclose(distort_fp);

	for (int index_offset = 0; index_offset <= PEACH_SEQUENCE_LENGTH; index_offset++) {

	  int seqnum = index - index_offset;
	  string seqfullpathname = fullpathname_from_logdir_and_seqnum(logdir, seqnum);
	  
	  seqim = smart_imread(seqfullpathname);

	  // DISTORT (including crop/resize)...

	  Mat distorted_output_im = apply_precomputed_distortion(V_distort[d],
								 seqim, 
								 FarXval, FarXval_new,
								 NearXval, NearXval_new);

	  // ...and write

	  image_ss.str("");
	  image_ss << ss.str() << "/" << setw(3) << setfill('0') << index_offset << ".jpg";

	  // write image_ss

	  imwrite(image_ss.str().c_str(), distorted_output_im);

	  // write to distorted_trainvert file

	  if (index_offset == 0) {

	    line_ss.str("");
	    line_ss << short_ss.str().c_str() << ", " << imsig.c_str() << ", " << FarXval_new[0] << ", " << FarXval_new[1] << ", " << NearXval_new[0] << ", " << NearXval_new[1];
	    Line_distorted[Idx_distorted[distort_index]] = line_ss.str();
	  }
	  
	}

	distort_index++;
      }
    }
  }

  printf("Line_distorted size %i\n", (int) Line_distorted.size());
  fflush(stdout);

  for (i = 0; i < Line_distorted.size(); i++) {
    fprintf(distorted_train_vert_fp, "%s\n", Line_distorted[i].c_str());
    fflush(distorted_train_vert_fp);
  }

  // close train/test vert files
  
  fclose(train_vert_fp);
  fclose(test_vert_fp);
  fclose(distorted_train_vert_fp);
}

//----------------------------------------------------------------------------

// top and bottom
// different crop
// with distortions

void write_traintest_strawberry_color(string dir, float training_fraction)
{
  int i, distort_index, num_training, x_new, y_new, x_left, y_top;
  Mat im;
  //  vector < pair <string, vector <Point> > > TrainTest;
  vector < pair <string, VertVect > > TrainTest;
  set<int>::iterator iter;
  stringstream ss, short_ss;
  string date_str;
  string train_path, distorted_train_path, test_path;
  float scale_factor = (float) output_width / (float) output_crop_width;    // e.g., 160 / 250

  filter_for_traintest();

  date_str = string(UD_datetime_string());

  ss << "mkdir " << dir << "_" << date_str;
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/train";
  train_path = ss.str();
  ss.str("");
  ss << "mkdir " << train_path; 
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/distorted_train";
  distorted_train_path = ss.str();
  ss.str("");
  ss << "mkdir " << distorted_train_path; 
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/test";
  test_path = ss.str();
  ss.str("");
  ss << "mkdir " << test_path; 
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  num_training = (int) rint(training_fraction * (float) FilteredVert_idx_set.size());
  x_left = CANONICAL_IMAGE_WIDTH / 2 - output_crop_width / 2;
  y_top = output_crop_top_y;

  // copy names from Filtered_idx_set into vector 
  
  TrainTest.clear();
  for (iter = FilteredVert_idx_set.begin(); iter != FilteredVert_idx_set.end(); iter++) 
    TrainTest.push_back(make_pair(Fullpathname_vect[*iter], Vert[*iter]));

  // shuffle 

  random_shuffle(TrainTest.begin(), TrainTest.end());

  // open train, test vert files

  string train_vert_filename;
  string distorted_train_vert_filename;
  string test_vert_filename;
  string params_filename;
  FILE *train_vert_fp, *distorted_train_vert_fp, *test_vert_fp, *params_fp;

  ss.str("");
  ss << dir << "_" << date_str << "/trainvert.txt";
  train_vert_filename = ss.str();
  train_vert_fp = fopen(train_vert_filename.c_str(), "w");

  ss.str("");
  ss << dir << "_" << date_str << "/distorted_trainvert.txt";
  distorted_train_vert_filename = ss.str();
  distorted_train_vert_fp = fopen(distorted_train_vert_filename.c_str(), "w");

  ss.str("");
  ss << dir << "_" << date_str << "/testvert.txt";
  test_vert_filename = ss.str();
  test_vert_fp = fopen(test_vert_filename.c_str(), "w");

  printf("%s\n%s\n", train_vert_filename.c_str(), test_vert_filename.c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/params.txt";
  params_filename = ss.str();
  params_fp = fopen(params_filename.c_str(), "w");

  fprintf(params_fp, "label type = FAR x left, FAR x right, NEAR x left, NEAR x right (ordered)\n");

  fprintf(params_fp, "crop width = %i\n", output_crop_width);
  fprintf(params_fp, "crop height = %i\n", output_crop_height);
  fprintf(params_fp, "crop top y = %i\n", output_crop_top_y);

  fprintf(params_fp, "output width = %i\n", output_width);
  fprintf(params_fp, "output height = %i\n", output_height);

  fprintf(params_fp, "similarity threshold = %.3f\n", SIMILARITY_THRESHOLD);
  fprintf(params_fp, "training fraction = %.3f\n", training_fraction);

  fprintf(params_fp, "input data = %s\n", s_imagedirs.c_str());

  fprintf(params_fp, "# raw images = %i (current)\n", (int) Fullpathname_vect.size());
  fprintf(params_fp, "# bad images = %i (current)\n", (int) Bad_idx_set.size());
  fprintf(params_fp, "# GT images = %i (current)\n", (int) Vert_idx_set.size());
  fprintf(params_fp, "# filtered images = %i (current)\n", (int) FilteredVert_idx_set.size());

  fclose(params_fp);

  // generate and shuffle indices of distorted training images
  
  vector <string> Line_distorted;
  vector <int> Idx_distorted;
  
  Line_distorted.resize(num_training * distort_num_per_image);
  Idx_distorted.resize(num_training * distort_num_per_image);
  for (int z = 0; z < Idx_distorted.size(); z++)
    Idx_distorted[z] = z;
  random_shuffle(Idx_distorted.begin(), Idx_distorted.end());

  // iterate through and load each image into "im"

  for (i = 0, distort_index = 0; i < TrainTest.size(); i++) {

    //    printf("new %i: reading %s\n", i, TrainTest[i].first.c_str());

    im = smart_imread(TrainTest[i].first);
    string imsig = fullpathname_to_signature(TrainTest[i].first);

    // for each, crop, resize, and WRITE
    
    Mat output_color_im = crop_and_scale_image(im, 
					       im.cols / 2 - output_crop_width / 2, im.cols / 2 + output_crop_width / 2,
					       output_crop_top_y, output_crop_height,
					       output_width, output_height);
    //    Mat output_color_im = crop_and_scale_image(im, output_crop_top_y, output_crop_width, output_crop_height, output_width, output_height);

    /*
    Rect crop_rect(im.cols / 2 - output_crop_width / 2, output_crop_top_y, 
		   output_crop_width, output_crop_height);
    Mat crop_im = im(crop_rect);
    Mat output_color_im = Mat(output_height, output_width, CV_8UC3);
    resize(crop_im, output_color_im, output_color_im.size(), 0, 0, INTER_CUBIC);
    //    Mat output_im = Mat(output_height, output_width, CV_8UC1);
    */

    //    cvtColor(output_cubic_im, output_im, cv::COLOR_RGB2GRAY);
  
    // WRITE NEW IMAGE, NEW VERTS WITH NEW NAME!
    
    string imname;
    FILE *fp;
    vector <int> FarXval, NearXval;

    ss.str("");
    short_ss.str("");

    // calculate new vert coords 
    
    for (int j = 0; j < TrainTest[i].second.size(); j++) {

      if (TrainTest[i].second[j].y == IMAGE_ROW_FAR) {

	x_new = (int) rint(scale_factor * (float) (TrainTest[i].second[j].x - x_left));
	y_new = (int) rint(scale_factor * (float) (TrainTest[i].second[j].y - y_top));

	//	printf("%i: %i, %i\n", j, x_new, y_new); fflush(stdout);
 
	// WRITE THEM

	//	if (!(j % 2)) {
	FarXval.push_back(x_new);
	  //}
	//else
	// fprintf(fp, "%i\n", x_new);
      }

      else if (TrainTest[i].second[j].y == IMAGE_ROW_NEAR) {

	x_new = (int) rint(scale_factor * (float) (TrainTest[i].second[j].x - x_left));
	y_new = (int) rint(scale_factor * (float) (TrainTest[i].second[j].y - y_top));

	//	printf("%i: %i, %i\n", j, x_new, y_new); fflush(stdout);
 
	// WRITE THEM

	//	if (!(j % 2)) {
	NearXval.push_back(x_new);
	  //}
	//else
	// fprintf(fp, "%i\n", x_new);
      }
    }

    sort(FarXval.begin(), FarXval.end());
    sort(NearXval.begin(), NearXval.end());

    // path info

    if (i < num_training) {

      generate_far_near_training_image_distortions(im, imsig, FarXval, NearXval,
						   distort_num_per_image, distort_index, distorted_train_path,
						   Idx_distorted,
						   Line_distorted,
						   distorted_train_vert_fp);

      //printf("TRAIN: ");
      ss << train_path << "/" << setw(6) << setfill('0') << i << ".jpg";
      short_ss << "train/" << setw(6) << setfill('0') << i << ".jpg";
      fp = train_vert_fp;
    }
    else {
      //printf("TEST: ");
      ss << test_path << "/" << setw(6) << setfill('0') << i - num_training << ".jpg";
      short_ss << "test/" << setw(6) << setfill('0') << i - num_training << ".jpg";
      fp = test_vert_fp;
    }

    imname = ss.str();

    //    printf("%s\n", imname.c_str());
    imwrite(imname.c_str(), output_color_im);

    //    fprintf(fp, "%s, %s, ", imname.c_str(), TrainTest[i].first.c_str());
    //    fprintf(fp, "%s, %s, ", short_ss.str().c_str(), TrainTest[i].first.c_str());

    //printf("writing %s\n", imsig.c_str()); fflush(stdout);
    fprintf(fp, "%s, %s, ", short_ss.str().c_str(), imsig.c_str());
    fflush(fp);

    // write new vert coords 

    //    line(output_color_im, Point(FarXval[0], y_new), Point(FarXval[1], y_new), Scalar(0, 255, 0), 2);

    fprintf(fp, "%i, %i, %i, %i\n", FarXval[0], FarXval[1], NearXval[0], NearXval[1]);
    fflush(fp);

    // wait to write image so we can draw on it for debugging

    // imwrite(imname.c_str(), output_cubic_im);
    
  }

  for (i = 0; i < Line_distorted.size(); i++) {
    fprintf(distorted_train_vert_fp, "%s\n", Line_distorted[i].c_str());
    fflush(distorted_train_vert_fp);
  }

  // close train/test vert files
  
  fclose(train_vert_fp);
  fclose(test_vert_fp);
  fclose(distorted_train_vert_fp);
}

//----------------------------------------------------------------------------

// ONLY WRITE NON-TRAIL (AKA "BAD") IMAGES 

// same crop/scale ("strawberry") as top and bottom 

void write_traintest_nontrail_color(string dir, float training_fraction)
{
  int i, distort_index, num_training;
  Mat im;
  //  vector < pair <string, vector <Point> > > TrainTest;
  vector < string > BadTrainTest;
  set<int>::iterator iter;
  stringstream ss, short_ss;
  string date_str;
  string train_path, distorted_train_path, test_path;
  float scale_factor = (float) output_width / (float) output_crop_width;    // e.g., 160 / 250

  filter_out_overly_similar_bad_images();

  date_str = string(UD_datetime_string());

  ss << "mkdir " << dir << "_" << date_str;
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/nontrail_train";
  train_path = ss.str();
  ss.str("");
  ss << "mkdir " << train_path; 
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/nontrail_distorted_train";
  distorted_train_path = ss.str();
  ss.str("");
  ss << "mkdir " << distorted_train_path; 
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/nontrail_test";
  test_path = ss.str();
  ss.str("");
  ss << "mkdir " << test_path; 
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  num_training = (int) rint(training_fraction * (float) FilteredBad_idx_set.size());

  // copy names from Filtered_idx_set into vector 
  
  BadTrainTest.clear();
  for (iter = FilteredBad_idx_set.begin(); iter != FilteredBad_idx_set.end(); iter++) 
    BadTrainTest.push_back(Fullpathname_vect[*iter]);

  // shuffle 

  random_shuffle(BadTrainTest.begin(), BadTrainTest.end());

  // open train, test vert files

  string train_filename, distorted_train_filename;
  string test_filename;
  string params_filename;
  FILE *train_fp, *distorted_train_fp, *test_fp, *params_fp;

  ss.str("");
  ss << dir << "_" << date_str << "/nontrail_train.txt";
  train_filename = ss.str();
  train_fp = fopen(train_filename.c_str(), "w");

  ss.str("");
  ss << dir << "_" << date_str << "/nontrail_distorted_train.txt";
  distorted_train_filename = ss.str();
  distorted_train_fp = fopen(distorted_train_filename.c_str(), "w");

  ss.str("");
  ss << dir << "_" << date_str << "/nontrail_test.txt";
  test_filename = ss.str();
  test_fp = fopen(test_filename.c_str(), "w");

  printf("%s\n%s\n", train_filename.c_str(), test_filename.c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/nontrail_params.txt";
  params_filename = ss.str();
  params_fp = fopen(params_filename.c_str(), "w");

  fprintf(params_fp, "label type = none -- all images are BAD (non-trail)\n");

  fprintf(params_fp, "crop width = %i\n", output_crop_width);
  fprintf(params_fp, "crop height = %i\n", output_crop_height);
  fprintf(params_fp, "crop top y = %i\n", output_crop_top_y);

  fprintf(params_fp, "output width = %i\n", output_width);
  fprintf(params_fp, "output height = %i\n", output_height);

  fprintf(params_fp, "similarity threshold = %.3f\n", SIMILARITY_THRESHOLD);
  fprintf(params_fp, "training fraction = %.3f\n", training_fraction);

  fprintf(params_fp, "input data = %s\n", s_imagedirs.c_str());

  fprintf(params_fp, "# raw images = %i (current)\n", (int) Fullpathname_vect.size());
  fprintf(params_fp, "# bad images = %i (current)\n", (int) Bad_idx_set.size());
  fprintf(params_fp, "# GT images = %i (current)\n", (int) Vert_idx_set.size());
  fprintf(params_fp, "# filtered bad images = %i (current)\n", (int) FilteredBad_idx_set.size());

  fclose(params_fp);

  // generate and shuffle indices of distorted training images

  vector <string> Line_distorted;
  vector <int> Idx_distorted;
  
  Line_distorted.resize(num_training * nontrail_distort_num_per_image);
  Idx_distorted.resize(num_training * nontrail_distort_num_per_image);
  for (int z = 0; z < Idx_distorted.size(); z++)
    Idx_distorted[z] = z;
  random_shuffle(Idx_distorted.begin(), Idx_distorted.end());

  // iterate through and load each image into "im"

  for (i = 0, distort_index = 0; i < BadTrainTest.size(); i++) {

    //    printf("new %i: reading %s\n", i, TrainTest[i].first.c_str());

    im = smart_imread(BadTrainTest[i]);
    string imsig = fullpathname_to_signature(BadTrainTest[i]);

    // for each, crop, resize, and WRITE
    
    Mat output_color_im = crop_and_scale_image(im, 
					       im.cols / 2 - output_crop_width / 2, im.cols / 2 + output_crop_width / 2,
					       output_crop_top_y, output_crop_height,
					       output_width, output_height);

    /*
    Rect crop_rect(im.cols / 2 - output_crop_width / 2, output_crop_top_y, 
		   output_crop_width, output_crop_height);
    Mat crop_im = im(crop_rect);
    Mat output_color_im = Mat(output_height, output_width, CV_8UC3);
    resize(crop_im, output_color_im, output_color_im.size(), 0, 0, INTER_CUBIC);
    //    Mat output_im = Mat(output_height, output_width, CV_8UC1);
    */

    //    cvtColor(output_cubic_im, output_im, cv::COLOR_RGB2GRAY);
  
    // WRITE NEW IMAGE WITH NEW NAME!
    
    string imname;
    FILE *fp;

    ss.str("");
    short_ss.str("");

    // path info

    if (i < num_training) {

      generate_nontrail_training_image_distortions(im, imsig, 
      						   nontrail_distort_num_per_image, distort_index, distorted_train_path,
      						   Idx_distorted,
      						   Line_distorted,
      						   distorted_train_fp);

      //printf("TRAIN: ");
      ss << train_path << "/" << setw(6) << setfill('0') << i << ".jpg";
      short_ss << "nontrail_train/" << setw(6) << setfill('0') << i << ".jpg";
      fp = train_fp;
    }
    else {
      //printf("TEST: ");
      ss << test_path << "/" << setw(6) << setfill('0') << i - num_training << ".jpg";
      short_ss << "nontrail_test/" << setw(6) << setfill('0') << i - num_training << ".jpg";
      fp = test_fp;
    }

    imname = ss.str();

    //    printf("%s\n", imname.c_str());
    imwrite(imname.c_str(), output_color_im);

    //    fprintf(fp, "%s, %s, ", imname.c_str(), TrainTest[i].first.c_str());
    //    fprintf(fp, "%s, %s, ", short_ss.str().c_str(), TrainTest[i].first.c_str());

    //printf("writing %s\n", imsig.c_str()); fflush(stdout);
    fprintf(fp, "%s, %s\n", short_ss.str().c_str(), imsig.c_str());
    fflush(fp);

    // wait to write image so we can draw on it for debugging

    // imwrite(imname.c_str(), output_cubic_im);
    
  }

  for (i = 0; i < Line_distorted.size(); i++) {
    fprintf(distorted_train_fp, "%s\n", Line_distorted[i].c_str());
    fflush(distorted_train_fp);
  }

  // close train/test vert files
  
  fclose(train_fp);
  fclose(test_fp);
  //  fclose(distorted_train_vert_fp);
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

// change which image we are currently processing/displaying

void set_current_index(int new_index)
{
  if (object_input_mode == SCALLOP_MODE) {

    int n = scallop_Fullpathname_vect.size();
        
    // wrap?
    
    if (new_index >= n)
      current_index = n - new_index;
    else if (new_index < 0)
      current_index = n + new_index;
    
    // normal
    
    else
      current_index = new_index;

  }
  else {
    
    editing_tree = false;
    p_tree_inter_result = false;
    p_tree_have_direction = false;
    
    int n = Fullpathname_vect.size();
    
    // if we changed image index mid-way through choosing vertices, CLEAR those vertices
    
    if (current_index < Vert.size() &&
	Vert[current_index].size() > 0 && Vert[current_index].size() != REQUIRED_NUMBER_OF_VERTS_PER_IMAGE) {
      Vert[current_index].clear();
    }
    
    // wrap?
    
    if (new_index >= n)
      current_index = n - new_index;
    else if (new_index < 0)
      current_index = n + new_index;
    
    // normal
    
    else
      current_index = new_index;
    
    // is this a "bad" image?
    
    bad_current_index = isBad(current_index);
    
    // do we have vertices for this image?
    
    vert_current_index = isVert(current_index);
  }
}

//----------------------------------------------------------------------------

// if not commented, get all jpg/png image names in directory dirname, append them to filenames string vector

void add_images(string dirname, vector <string> & imnames)
{
  string imname;
  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir (dirname.c_str())) != NULL) {
    cout << "loading " << dirname << endl;
    while ((ent = readdir (dir)) != NULL) {
      if (strstr(ent->d_name, "jpg") != NULL || strstr(ent->d_name, "png") != NULL) {
	
	imname = dirname + string(ent->d_name);
	imnames.push_back(imname);
	//	printf("read %s\n", imname.c_str());


      }
    }
    closedir (dir);
  } 
}

//----------------------------------------------------------------------------

// *each line* of file loaded here should have the full path to a directory full of numbered images 

// this function should be run ONCE, before anything else happens

void add_all_images_from_file(string impathfilename)
{
  ifstream inStream;
  string line;

  // read file

  inStream.open(impathfilename.c_str());

  if (!inStream.good()) {
    printf("problem opening %s -- skipping\n", impathfilename.c_str());
    return;
  }

  while (getline(inStream, line)) 
    add_images(line, Fullpathname_vect);

  // get them in some sort of order

  sort(Fullpathname_vect.begin(), Fullpathname_vect.end());

  // make maps

  Idx_signature_vect.resize(Fullpathname_vect.size());

  for (int i = 0; i < Fullpathname_vect.size(); i++) {
    string sig = fullpathname_to_signature(Fullpathname_vect[i]);
    Signature_idx_map[sig] = i;
    Idx_signature_vect[i] = sig;
  }
}

//----------------------------------------------------------------------------

// return closest guide row

int snap_y(int y)
{
  int diff, min_diff, min_y;
  min_diff = 10000;

  for (int i = 0; i < trailEdgeRow.size(); i++) {
    diff = abs(y - trailEdgeRow[i]);
    if (diff < min_diff) {
      min_diff = diff;
      min_y = trailEdgeRow[i];
    }
  }

  return min_y;
}

//----------------------------------------------------------------------------

void print_treeparams(FILE *fp, TreeParams & t)
{
  fprintf(fp, "%i, %i, %.3f, %.3f, %.2f\n",
	  t.v_bottom.x, t.v_bottom.y,
	  t.dx, t.dy,
	  t.width);
}

//----------------------------------------------------------------------------

// SCALLOP MODE: what to do when mouse is moved/mouse button is push/released

void scallop_onMouse(int event, int x, int y, int flags, void *userdata)
{
  map<int, string>::iterator iter;
  string imsig;

  getScallopSignature(current_imname, imsig);

  // dragging...

  if  ( event == EVENT_MOUSEMOVE ) {

    if (dragging) {

      p_scallop_end.x = x;
      p_scallop_end.y = y;

    }
    else {
      p_mouse.x = x;
      p_mouse.y = y;

      p_mouse_dx.x = x + 100;
      p_mouse_dx.y = y;

      p_mouse_dy.x = x;
      p_mouse_dy.y = y + 100;
    }
    
  }

  // if we are inside a scallop bbox, cancel it

  else if  ( event == EVENT_RBUTTONDOWN ) {

    // iterate through all scallops in this image...
    
    for (int scallop_idx = 0; scallop_idx < scallop_params_vect.size(); scallop_idx++) {

      iter = scallop_idx_signature_map.find(scallop_idx);

      if (iter != scallop_idx_signature_map.end() && (*iter).second == imsig) {

	// if we have a scallop bbox, check if click is inside it

	if (scallop_params_vect[scallop_idx].has_scale && 
	    x >= scallop_params_vect[scallop_idx].p_upper_left.x && x <= scallop_params_vect[scallop_idx].p_lower_right.x &&
	    y >= scallop_params_vect[scallop_idx].p_upper_left.y && y <= scallop_params_vect[scallop_idx].p_lower_right.y) {
	  scallop_params_vect[scallop_idx].has_scale = false;
	  break;
	}
      }
    }
  }
  
  // initiating segment

  else if  ( event == EVENT_LBUTTONDOWN ) {

    dragging = true;

    p_scallop_start.x = p_scallop_end.x = x;
    p_scallop_start.y = p_scallop_end.y = y;
  }

    // finishing segment

  else if  ( event == EVENT_LBUTTONUP ) {

    dragging = false;

    p_scallop_end.x = x;
    p_scallop_end.y = y;

    // now figure out which p_annotation this goes with
    // for each scallop in this image, see if p_annotation is "inside" p_scallop_* rect
    // if so, "associate" it.  if nothing inside, then this is a spurious rect
    
    // iterate through all scallops...
    
    for (int scallop_idx = 0; scallop_idx < scallop_params_vect.size(); scallop_idx++) {

      // ...and see if *this* scallop corresponds to signature of current_imname
      
      iter = scallop_idx_signature_map.find(scallop_idx);

      // if so, check if it's inside scallop we are drawing

      if (iter != scallop_idx_signature_map.end() && (*iter).second == imsig) {
	Point p_ann = scallop_params_vect[scallop_idx].p_annotation;
	if (p_ann.x >= p_scallop_start.x && p_ann.x <= p_scallop_end.x &&
	    p_ann.y >= p_scallop_start.y && p_ann.y <= p_scallop_end.y) {
	  scallop_params_vect[scallop_idx].p_upper_left = p_scallop_start;
	  scallop_params_vect[scallop_idx].p_lower_right = p_scallop_end;
	  scallop_params_vect[scallop_idx].has_scale = true;
	}
	
	//	printf("%i, %i\n", scparams.p_annotation.x, scparams.p_annotation.y);
	//Point ul = Point(scparams.p_annotation.x - side/2, scparams.p_annotation.y - side/2);
	//Point lr = Point(scparams.p_annotation.x + side/2, scparams.p_annotation.y + side/2);
	
	//rectangle(draw_im, ul, lr, Scalar(0, 255, 0), 2);

      }
    }

  }
  
  draw_im = current_im.clone();
  scallop_draw_overlay();

  // show current edit that is underway
  
  if (dragging) {

    // draw some stuff

    rectangle(draw_im, p_scallop_start, p_scallop_end, Scalar(255, 0, 255), 1);

  }
  
  imshow("trailGT", draw_im);  

}

//----------------------------------------------------------------------------

// TREE MODE: what to do when mouse is moved/mouse button is push/released

// "rectangular ray": * first click is center of bottom edge (aka
// origin) * direction from first click to current location defines
// orientation of long axis -- THERE IS NO LENGTH -- THIS CORRESPONDS
// TO TOP OF TREE NOT BEING VISIBLE
// * distance from first click to current location is proportional to length of minor axis (aka width at *base -- SINCE UNDER PERSPECTIVE IMAGE "WIDTH" OF TRUNK WILL INCREASE AS WE GO UP TRUNK)

void tree_onMouse(int event, int x, int y, int flags, void *userdata)
{
Point p_width;
  int g, b;
  
  // dragging...

  if  ( event == EVENT_MOUSEMOVE ) {

    // if dragging, we must be editing tree
    
    if (dragging) {

      p_tree_upper.x = x;
      p_tree_upper.y = y;  

      p_tree_inter_result = ray_image_boundaries_intersection(p_tree_bottom, p_tree_upper, RAY_DELTA_THRESH, p_tree_inter);
      
      // set line color to indicate state

      g = 200;
      b = 0;

    }

    // not dragging
    
    else {

      // set line color to indicate state
      
      g = 255; 
      b = 255;
      
      if (editing_tree) {

	      // we have a valid trunk direction
      
	if (p_tree_inter_result) {
	  tree_dx_ortho = p_tree_upper.x - p_tree_bottom.x;
	  tree_dy_ortho = p_tree_upper.y - p_tree_bottom.y;
	  double temp = tree_dx_ortho;
	  tree_dx_ortho = -tree_dy_ortho;
	  tree_dy_ortho = temp;
	  double len = sqrt(tree_dx_ortho*tree_dx_ortho + tree_dy_ortho*tree_dy_ortho);
	  tree_dx_ortho /= len;
	  tree_dy_ortho /= len;
	  
	  p_tree_have_direction = true;
	}

	p_width.x = x;
	p_width.y = y;

	// distance from upper point
	//p_tree_width_val = (double) ((p_width.x - p_tree_upper.x)*(p_width.x - p_tree_upper.x) + (p_width.y - p_tree_upper.y)*(p_width.y - p_tree_upper.y));
	//p_tree_width_val = sqrt(p_tree_width_val);

	// distance from line between bottom and upper point
	p_tree_width_val = fabs(point_line_distance(p_width, p_tree_bottom, p_tree_upper));
	
	//	printf("dist %.3lf\n", p_tree_width_val);
      }
      else
	p_tree_have_direction = false;

    }

  }
  
  // cancel

  else if  ( event == EVENT_RBUTTONDOWN ) {
    editing_tree = false;
    p_tree_inter_result = false;
    p_tree_have_direction = false;

  }
  
  // initiating segment

  else if  ( event == EVENT_LBUTTONDOWN ) {

    // ignore initial clicks outside of FAR-NEAR y range
    
    //    if (y >= IMAGE_ROW_FAR && y <= IMAGE_ROW_NEAR) {
    if (!editing_tree)
      editing_tree = true;
    else {
      // finished!

      TreeParams t;
      t.v_bottom = p_tree_bottom;
      t.dx = tree_dy_ortho;
      t.dy = -tree_dx_ortho;
      t.width = 2.0 * p_tree_width_val;
      print_treeparams(stdout, t);

      printf("inserting tree at image %i\n", current_index);
      Tree_idx_set.insert(current_index);

      editing_tree = false;
      p_tree_have_direction = false;
      p_tree_inter_result = false;
    }
    
    dragging = true;
    
    g = 255;
    b = 0;
    p_tree_upper.x = x;
    p_tree_upper.y = y; // IMAGE_ROW_FAR;
    //      vboundary.x = x;
    //vboundary.y = 0;
    p_tree_bottom.x = x;
    p_tree_bottom.y = y;
    // }
  }

  // finishing segment

  else if  ( event == EVENT_LBUTTONUP ) {

    dragging = false;

    g = 255;
    b = 0;

    if (!p_tree_inter_result) {
      editing_tree = false;
    }
    
    // normal
    //    if (y <= v_initial.y) {
      p_tree_upper.x = x;
      p_tree_upper.y = y;  // IMAGE_ROW_FAR;
      //}
    // mirror
    //else if (y > v_initial.y) {
    //  v.x = v_initial.x - (x - v_initial.x);
    //  v.y = v_initial.y - (y - v_initial.y);  // IMAGE_ROW_FAR;
    //}

    //    v.x = x;
    // v.y = y;  // IMAGE_ROW_FAR;

  }

  draw_im = current_im.clone();
  tree_draw_overlay();

  // show current edit that is underway
  
  //  if (dragging) {
  if (editing_tree) {
    line(draw_im, p_tree_bottom, p_tree_upper, Scalar(0, 255, 255), 1);
    if (p_tree_inter_result)
      line(draw_im, p_tree_upper, p_tree_inter, Scalar(255, 255, 0), 1);
    if (p_tree_have_direction) {
      int idx = (int) rint(p_tree_width_val * tree_dx_ortho);
      int idy = (int) rint(p_tree_width_val * tree_dy_ortho);
      //      printf("%lf\n", p_tree_width_val);
      Point p_bottom_right = Point(p_tree_bottom.x + idx, p_tree_bottom.y + idy);
      Point p_upper_right = Point(p_tree_upper.x + idx, p_tree_upper.y + idy);
      line(draw_im, p_tree_bottom, p_bottom_right, Scalar(0, 0, 255), 1);
      line(draw_im, p_upper_right, p_bottom_right, Scalar(0, 0, 255), 1);
      Point p_bottom_left = Point(p_tree_bottom.x - idx, p_tree_bottom.y - idy);
      Point p_upper_left = Point(p_tree_upper.x - idx, p_tree_upper.y - idy);
      line(draw_im, p_tree_bottom, p_bottom_left, Scalar(255, 0, 0), 1);
      line(draw_im, p_upper_left, p_bottom_left, Scalar(255, 0, 0), 1);

      Point p_right_inter, p_left_inter;
      
      if (ray_image_boundaries_intersection(p_bottom_right, p_upper_right, RAY_DELTA_THRESH, p_right_inter))
	line(draw_im, p_upper_right, p_right_inter, Scalar(255, 0, 255), 1);

      if (ray_image_boundaries_intersection(p_bottom_left, p_upper_left, RAY_DELTA_THRESH, p_left_inter))
	line(draw_im, p_upper_left, p_left_inter, Scalar(255, 0, 255), 1);

    }
  }
  
  // where is cursor?

//  if (!erasing)
//    circle(draw_im, Point (v.x, v.y), 8, Scalar(0, g, b), 1, 8, 0);

  imshow("trailGT", draw_im);  

}

//----------------------------------------------------------------------------

// TRAIL MODE: what to do when mouse is moved/mouse button is push/released

void trail_onMouse(int event, int x, int y, int flags, void *userdata)
{
  Point v;
  int g, b;

  // no mouse interaction if image being or already marked bad

  if (do_bad || bad_current_index) 
    return;

  // dragging...

  if  ( event == EVENT_MOUSEMOVE ) {

    //    vx = x;
    v.x = x;
    if (dragging) {
      g = 200;
      b = 0;
      v.y = dragging_y;
    }
    else {
      g = 255; 
      b = 255;
      v.y = snap_y(y);
    }
  }

  // initiating horizontal segment

  else if  ( event == EVENT_LBUTTONDOWN ) {

    dragging = true;

    g = 255;
    b = 0;
    v.x = x;
    v.y = snap_y(y);
    dragging_x = v.x;
    dragging_y = v.y;

    // clear any existing verts *on this row*

    //    vector <Point>::iterator iter = Vert[current_index].begin();
    VertVect::iterator iter = Vert[current_index].begin();

    while (iter != Vert[current_index].end()) {
      if ((*iter).y == v.y)
	iter = Vert[current_index].erase(iter);
      else
	iter++;
    }

    Vert[current_index].push_back(v);

  }

  // finishing horizontal segment

  else if  ( event == EVENT_LBUTTONUP ) {

    dragging = false;

    g = 255;
    b = 0;
    v.x = x;
    v.y = dragging_y;

    Vert[current_index].push_back(v);

    // done with this image!

    if (Vert[current_index].size() == REQUIRED_NUMBER_OF_VERTS_PER_IMAGE) {

      Vert_idx_set.insert(current_index);

      set<int>::iterator iter = NoVert_idx_set.find(current_index);
      // if it is actually in the set, erase it
      if (iter != NoVert_idx_set.end())
	NoVert_idx_set.erase(iter);

      vert_current_index = true;

      if (next_nonvert_idx != NO_INDEX)
	next_nonvert_idx = most_isolated_nonvert_image_idx(current_index);
    }

  }

  // clear vertices for this image

  else if  ( event == EVENT_RBUTTONDOWN ) {

    erasing = true;

  }

  // clear vertices for this image

  else if  ( event == EVENT_RBUTTONUP ) {

    erasing = false;

    Vert[current_index].clear();

    set<int>::iterator iter = Vert_idx_set.find(current_index);
    // if it is actually in the set, erase it
    if (iter != Vert_idx_set.end())
      Vert_idx_set.erase(iter);

    NoVert_idx_set.insert(current_index);

    vert_current_index = false;

    // this could be done more efficiently, but it should be a pretty rare event
    if (next_nonvert_idx != NO_INDEX)
      next_nonvert_idx = most_isolated_nonvert_image_idx();
  }

  draw_im = current_im.clone();
  trail_draw_overlay();

  // show current edit that is underway

  if (dragging)
    line(draw_im, Point(dragging_x, v.y), Point(v.x, v.y), Scalar(0, 255, 255), 2);

  // where is cursor?

  if (!erasing)
    circle(draw_im, Point (v.x, v.y), 8, Scalar(0, g, b), 1, 8, 0);

  imshow("trailGT", draw_im);  
}

//----------------------------------------------------------------------------

// what to do when a key is pressed

void onKeyPress(char c, bool print_help)
{
  int idx;
  int step_idx = 1;

  // goto image 0 in sequence
  
  if (c == '0' || print_help) {

    if (print_help) 
      printf("0 = return to index 0 image\n");
    else
      set_current_index(ZERO_INDEX);

  }

  // most isolated nonvert image remaining

  if (c == 'n' || print_help) {

    if (print_help) 
      printf("n = save vertmap and goto most isolated nonvert image remaining\n");
    else if (next_nonvert_idx != NO_INDEX) {
      saveVertMap();   // for speed
      set_current_index(next_nonvert_idx);
    }
    else {
      printf("most isolated vertex not active\n");
    }
  }

  // filter out overly-similar images

  if (c == 'F' || print_help) {

    if (print_help) 
      printf("F = filter for traintest\n");
    else
      filter_for_traintest();

  }

  // goto next image in sequence
  
  if (c == 'x' || c == 'X' || print_help) {

    if (print_help)
      printf("x = goto next image, X = take bigger step to next image\n");
    else {
      if (c == 'X')
	step_idx = BIG_INDEX_STEP;
      
      if (!do_random) {
	
	if (do_verts) {
	  set<int>::iterator iter = Vert_idx_set.find(current_index);
	  if ((iter != Vert_idx_set.end()) && (iter == --Vert_idx_set.end()))
	    iter = Vert_idx_set.begin();
	  else
	    iter++;
	  set_current_index(*iter);
	}
	else
	  set_current_index(current_index + step_idx);
	
      }
      else {
	idx = Nonrandom_idx[current_index] + step_idx;  
	
	if (idx >= Fullpathname_vect.size())
	  idx = 0;
	
	set_current_index(Random_idx[idx]);
      }
    }
  }
  
  // goto previous image in sequence
  
  if (c == 'z' || c == 'Z' || print_help) {
    
    if (print_help) 
      printf("z = goto previous image, Z = take bigger step to previous image\n");
    else {
      if (c == 'Z')
	step_idx = BIG_INDEX_STEP;
      
      if (!do_random) {
	
	if (do_verts) {
	  set<int>::iterator iter = Vert_idx_set.find(current_index);
	  if (iter != Vert_idx_set.begin())
	    iter--;
	  else {
	    iter = Vert_idx_set.end();
	    --iter;
	  }
	  set_current_index(*iter);
	  
	}
	else
	  set_current_index(current_index - step_idx);
	
      }
      else {
	idx = Nonrandom_idx[current_index] - step_idx;  
	
	if (idx < 0)
	  idx = Fullpathname_vect.size() - 1;
	
	set_current_index(Random_idx[idx]);
      }   
    }
  }
  
  // toggle showing crop rectangle

  if (c == 'c' || print_help) {
    if (print_help)
      printf("c = toggle show crop rect\n");
    else
      do_show_crop_rect = !do_show_crop_rect;
  }

  // toggle overlay
  
  if (c == 'o' || print_help) {
    if (print_help)
      printf("o = toggle show overlay\n");
    else
      do_overlay = !do_overlay;
  }

  // toggle randomized index mode
  
  if (c == 'r' || print_help) {
    if (print_help)
      printf("r = toggle randomized index mode\n");
    else {
      if (!do_verts)
	do_random = !do_random;
    }
  }

  // toggle vert image-only mode
  
  if (c == 'v' || print_help) {
    if (print_help) 
      printf("v = toggle vert image only mode\n");
    else {
      if (vert_current_index) {
	do_verts = !do_verts;
	if (do_verts)
	  do_random = false;
      }
      else {
	printf("must be on a trail vert image to toggle verts-only mode\n");
      }
    }
  }

  // this is a bad image (no trail or trail geometry is wacky)
  
  if (c == 'b' || print_help) {
    if (print_help) 
      printf("b = mark this image as bad\n");
    else {
      if (do_random)
	return;
      do_bad = !do_bad;
      if (do_bad)
	bad_start = current_index;
      else {
	bad_end = current_index;
	for (int i = bad_start; i <= bad_end; i++)
	  Bad_idx_set.insert(i);
      }
    }
  }

  // allow bad images to become good again
  
  if (c == 'g' || print_help) {
    if (print_help)
      printf("g = mark this (bad) image as good\n");
    else {
      set<int>::iterator iter;
      iter = Bad_idx_set.find(current_index);
      if (iter != Bad_idx_set.end())
	Bad_idx_set.erase(iter);
      bad_current_index = false;
    }
  }

  // save everything

  if (c == 'S' || print_help) {
    if (print_help) {
      printf("s = save maps\n");   // e.g., bad, vert
    }
    else {

      if (object_input_mode == TRAIL_MODE) {
	saveBadMap();
	saveVertMap();
      }
      else if (object_input_mode == SCALLOP_MODE) {
	saveScallopMap();
      }
    }
  }

  if (c == 'j') {

    if (object_input_mode == SCALLOP_MODE) {
      imwrite("prediction.png", draw_im);
    }
  }
  
  // write traintest

  if (c == 'T' || print_help) {
    if (print_help)
      printf("T = write traintest for localization\n");
    else {
      if (object_input_mode == TRAIL_MODE) {
	//    write_traintest_vanilla_grayscale();
#ifdef OUTPUT_CHOCOLATE_DATA
	write_traintest_chocolate_color();
#elif defined(OUTPUT_STRAWBERRY_DATA)
	write_traintest_strawberry_color();
#elif defined(OUTPUT_PEACH_DATA)
	write_traintest_peach_color();
#else
	printf("undefined output data type\n");
	exit(1);
#endif
      }
      else if (object_input_mode == SCALLOP_MODE) {
	write_traintest_scallop();
      }
    }
  }

    // compute stats

  if (c == 'Y' || print_help) {
    if (print_help)
      printf("Y = compute stats\n");
    else {
      if (object_input_mode == SCALLOP_MODE) 
	compute_stats_traintest_scallop();
    }
  }

  // write non-trail

  if (c == 'N' || print_help) {
    if (print_help)
      printf("N = write traintest for trail/not-trail\n");
    else 
      write_traintest_nontrail_color();
  }

  // go to TRAIL mode

  if (c == '1' || print_help) {
    if (print_help)
      printf("1 = change to TRAIL input mode\n");
    else if (object_input_mode != TRAIL_MODE) {
      printf("switching to TRAIL mode!\n");
      object_input_mode = TRAIL_MODE;
      setMouseCallback("trailGT", trail_onMouse);
    }
  }
  
  // go to TREE mode

  if (c == '2' || print_help) {
    if (print_help)
      printf("2 = change to TREE input mode\n");
    else if (object_input_mode != TREE_MODE) {
      printf("switching to TREE mode!\n");
      object_input_mode = TREE_MODE;
      setMouseCallback("trailGT", tree_onMouse);
    }
  }

  // go to SCALLOP mode

  if (c == '3' || print_help) {
    if (print_help)
      printf("3 = change to SCALLOP input mode\n");
    else if (object_input_mode != SCALLOP_MODE) {
      printf("switching to SCALLOP mode!\n");
      object_input_mode = SCALLOP_MODE;
      setMouseCallback("trailGT", scallop_onMouse);
    }
  }

}

//----------------------------------------------------------------------------

// set all mat values at given channel to given value

// from: http://stackoverflow.com/questions/23510571/how-to-set-given-channel-of-a-cvmat-to-a-given-value-efficiently-without-chang

void setChannel(Mat &mat, unsigned int channel, unsigned char value)
{
  // make sure have enough channels
  if (mat.channels() < channel + 1)
    return;
  
  const int cols = mat.cols;
  const int step = mat.channels();
  const int rows = mat.rows;
  for (int y = 0; y < rows; y++) {
    // get pointer to the first byte to be changed in this row
    unsigned char *p_row = mat.ptr(y) + channel; 
    unsigned char *row_end = p_row + cols*step;
    for (; p_row != row_end; p_row += step)
      *p_row = value;
  }
}

//----------------------------------------------------------------------------

void trail_draw_other_windows()
{
  // actually crop, scale, and save

  if (do_show_crop_rect) {
    
    Rect crop_rect(current_im.cols / 2 - output_crop_width / 2, output_crop_top_y, 
		   output_crop_width, output_crop_height);
    Mat crop_im = current_im(crop_rect);
    Mat output_cubic_im = Mat(output_height, output_width, CV_8UC3);
    Mat output_area_im = Mat(output_height, output_width, CV_8UC3);
    resize(crop_im, output_cubic_im, output_cubic_im.size(), 0, 0, INTER_CUBIC);
    resize(crop_im, output_area_im, output_area_im.size(), 0, 0, INTER_AREA);
    Mat output_im = Mat(output_height, output_width, CV_8UC1);
    //imwrite("crop.jpg", crop_im);
    //      imshow("crop", crop_im);

    cvtColor(output_area_im, output_im, cv::COLOR_RGB2GRAY);
    imshow("cubic", output_cubic_im);
    imshow("area", output_area_im);
    imshow("output (area)", output_im);
    char c = waitKey(5);
    
  }
}

//----------------------------------------------------------------------------


void scallop_draw_overlay()
{
  stringstream ss;
  int scallop_idx;
  map<int, string>::iterator iter;
  string imsig;
  int side = 10;
  
  if (do_overlay) {

    // draw upper-left "corner guide"?
    
    if (!dragging) {
      line(draw_im, p_mouse, p_mouse_dx, Scalar(0, 128, 128), 1);
      line(draw_im, p_mouse, p_mouse_dy, Scalar(0, 128, 128), 1);
    }
    
    // which image is this?

    getScallopSignature(current_imname, imsig);

    ss << "SCALLOP " << current_index << ": " << imsig; // current_imname;
    string str = ss.str();

    //    printf("%s\n", str.c_str());
    
    putText(draw_im, str, Point(5, 10), FONT_HERSHEY_SIMPLEX, fontScale, Scalar::all(255), 1, 8);
    
    // iterate through all scallops...
    
    for (scallop_idx = 0; scallop_idx < scallop_params_vect.size(); scallop_idx++) {

      // ...and see if *this* scallop corresponds to signature of current_imname
      
      iter = scallop_idx_signature_map.find(scallop_idx);

      // if so, draw it

      if (iter != scallop_idx_signature_map.end() && (*iter).second == imsig) {
	ScallopParams scparams = scallop_params_vect[scallop_idx];
	//printf("%i, %i\n", scparams.p_annotation.x, scparams.p_annotation.y);

	Point ul = Point(scparams.p_annotation.x - side/2, scparams.p_annotation.y - side/2);
	Point lr = Point(scparams.p_annotation.x + side/2, scparams.p_annotation.y + side/2);
	
	rectangle(draw_im, ul, lr, Scalar(0, 255, 0), 2);

	if (scparams.has_scale) {
	  rectangle(draw_im, scparams.p_upper_left, scparams.p_lower_right, Scalar(128, 0, 128), 2);
	}
	if (filter_for_traintest_scallop(scparams)) {
	  rectangle(draw_im, scparams.p_upper_left, scparams.p_lower_right, Scalar(255, 0, 255), 3);
	}
      }
    }
    //    printf("\n");
    // horizontal lines for trail edge rows
    
    //    for (int i = 0; i < trailEdgeRow.size(); i++) 
    //    line(draw_im, Point(0, trailEdgeRow[i]), Point(current_im.cols - 1, trailEdgeRow[i]), Scalar(0, 128, 128), 1);

  }

}

//----------------------------------------------------------------------------

void tree_draw_overlay()
{
  stringstream ss;
  
  if (do_overlay) {
    
    // which image is this?

    ss << "TREE " << current_index << ": " << current_imname;
    string str = ss.str();

    putText(draw_im, str, Point(5, 10), FONT_HERSHEY_SIMPLEX, fontScale, Scalar::all(255), 1, 8);

    // are we in "verts only" mode?

    if (do_verts) 
      putText(draw_im, "V", Point(25, 40), FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 255), 1, 8);

    // horizontal lines for trail edge rows
    
    for (int i = 0; i < trailEdgeRow.size(); i++) 
      line(draw_im, Point(0, trailEdgeRow[i]), Point(current_im.cols - 1, trailEdgeRow[i]), Scalar(0, 128, 128), 1);

  }
}

//----------------------------------------------------------------------------

// draw strings and shapes on top of current image

void trail_draw_overlay()
{
  stringstream ss;

  if (do_overlay) {
    
    // which image is this?

    ss << "TRAIL " << current_index << ": " << current_imname;
    string str = ss.str();

    putText(draw_im, str, Point(5, 10), FONT_HERSHEY_SIMPLEX, fontScale, Scalar::all(255), 1, 8);

    // isolation stats

    if (!bad_current_index && !vert_current_index) {
      ss.str("");
      ss << "max dist = " << max_closest_vert_dist << ", this dist = " << ClosestVert_dist[current_index];
      putText(draw_im, ss.str(), Point(5, 25), FONT_HERSHEY_SIMPLEX, fontScale, Scalar(255, 255, 255), 1, 8);
    }

    // save status

    int num_unsaved = Vert_idx_set.size() - num_saved_verts;

    if (num_unsaved > 0) {
      ss.str("");
      if (num_unsaved == 1)
	ss << num_unsaved << " image with unsaved verts [" << Vert_idx_set.size() << "]";
      else
	ss << num_unsaved << " images with unsaved verts [" << Vert_idx_set.size() << "]";
      str = ss.str();
      putText(draw_im, str, Point(5, 315), FONT_HERSHEY_SIMPLEX, 1.5 * fontScale, Scalar(0, 0, 255), 1, 8);
    }

    // show crop rectangle:

    if (do_show_crop_rect) {

      int center_x = draw_im.cols / 2;
      int xl = center_x - output_crop_width/2;
      int xr = center_x + output_crop_width/2;
      int yt = output_crop_top_y;
      int yb = output_crop_top_y + output_crop_height;

      rectangle(draw_im,  
		Point(xl, yt),
		Point(xr, yb),
		Scalar(255, 255, 255), 2);
      
    }

    // are we in "random next image" mode?

    if (do_random) 
      putText(draw_im, "R", Point(5, 40), FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 255), 1, 8);

    // are we in "bad image" marking mode?

    if (do_bad) 
      putText(draw_im, "B", Point(15, 40), FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 255), 1, 8);

    // are we in "verts only" mode?

    if (do_verts) 
      putText(draw_im, "V", Point(25, 40), FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 255), 1, 8);

    // is this a "bad" image?

    if (bad_current_index) {
      setChannel(draw_im, 2, 255);
      return;
    }

    // is this an image for which we have ground-truth trail edges?

    else if (vert_current_index) 
      setChannel(draw_im, 0, 200);
   
    // horizontal lines for trail edge rows

    for (int i = 0; i < trailEdgeRow.size(); i++) 
      line(draw_im, Point(0, trailEdgeRow[i]), Point(current_im.cols - 1, trailEdgeRow[i]), Scalar(0, 128, 128), 1);

    // trail edge vertices

    int r, g;

    if (erasing) {
      r = 255; g = 0;
    }
    else {
      r = 0; g = 255;
    }

    for (int i = 0; i < Vert[current_index].size(); i += 2) {

      // only draw line segment if we have a PAIR of verts (this will NOT be the case while a new segment is being drawn) 

      if (i + 1 < Vert[current_index].size())
	line(draw_im, Vert[current_index][i], Vert[current_index][i + 1], Scalar(0, g, r), 2);
    }

  }
}

//----------------------------------------------------------------------------

// write signatures for images which have been marked as "bad" -- i.e., they contain no trail
// don't write anything for non-bad images

void saveBadMap()
{    
  set<int>::iterator idx_iter;
  int i;

  FILE *fp = fopen("badmap.txt", "w");

  // new stuff

  for (idx_iter = Bad_idx_set.begin(); idx_iter != Bad_idx_set.end(); idx_iter++) {
    fprintf(fp, "%s\n", Idx_signature_vect[*idx_iter].c_str());
    fflush(fp);
  }

  // old stuff

  for (i = 0; i < External_bad_sig_vect.size(); i++) {
    fprintf(fp, "%s\n", External_bad_sig_vect[i].c_str());
    fflush(fp);
  }

  fclose(fp);
}

//----------------------------------------------------------------------------

// read file that specifies which images should be ignored because they contain
// no trail or the trail geometry does not conform to assumptions

// this assumes image filenames have already been loaded, so we know how many lines
// should be in this file

int loadBadMap()
{
  string sig;
  map<string, int>::iterator iter;
  ifstream inStream;
  int total = 0;

  inStream.open("badmap.txt");

  while (getline(inStream, sig)) {

    iter = Signature_idx_map.find(sig);

    // it's in the current dataset...

    if (iter != Signature_idx_map.end()) 
      Bad_idx_set.insert((*iter).second);

    // it's NOT -- but store it so it's saved properly later

    else
      External_bad_sig_vect.push_back(sig);

    total++;
  }

  inStream.close();

  return total;
}

//----------------------------------------------------------------------------

// check whether a given image is contained in current set of "bad" (non-trail") images

bool isBad(int idx)
{
  return (Bad_idx_set.find(idx) != Bad_idx_set.end());
}

//----------------------------------------------------------------------------

// check whether a given image is contained in set of those with ground-truth vertices
// specified

bool isVert(int idx)
{
  return (Vert_idx_set.find(idx) != Vert_idx_set.end());
}

//----------------------------------------------------------------------------

// write vector of vertices to filestream
// if this was smart, it could sort so that they are top to bottom, left to right or some such

void saveVertVect(FILE *fp, VertVect & V)
{
  int j;

  for (j = 0; j < V.size() - 1; j++)
    fprintf(fp, "(%i, %i), ", V[j].x, V[j].y);
  fprintf(fp, "(%i, %i)\n", V[j].x, V[j].y);
  fflush(fp);
}

//----------------------------------------------------------------------------

void saveVertVectNoParens(FILE *fp, VertVect & V)
{
  int j;

  for (j = 0; j < V.size() - 1; j++)
    fprintf(fp, "%i, %i, ", V[j].x, V[j].y);
  fprintf(fp, "%i, %i\n", V[j].x, V[j].y);
  fflush(fp);
}

//----------------------------------------------------------------------------

// write all current ground-truth trail vertices to file, **mapped by filename signature rather than index**
// filename signature should start with DATETIME directory

// requires that an image have exactly 4 verts to be written

void saveVertMap()
{
  int i, j;
  set<int>::iterator idx_iter;
  VertVect V;
  string sig;
  
  FILE *fp = fopen("vertmap.txt", "w");

  // new vects

  for (idx_iter = Vert_idx_set.begin(), num_saved_verts = 0; idx_iter != Vert_idx_set.end(); idx_iter++) {
    i = *idx_iter;
    if (Vert[i].size() == REQUIRED_NUMBER_OF_VERTS_PER_IMAGE) {
      fprintf(fp, "%s, ", Idx_signature_vect[i].c_str());
      saveVertVect(fp, Vert[i]);
      num_saved_verts++;
    }
    else if (Vert[i].size() > 0)
      printf("improper number of vertices for image %s; not saving\n", Idx_signature_vect[i].c_str());
  }

  // old vects

  for (i = 0; i < External_vert_vect.size(); i++) {
    sig = External_vert_vect[i].first;
    V = External_vert_vect[i].second;
    if (V.size() == REQUIRED_NUMBER_OF_VERTS_PER_IMAGE) {
      fprintf(fp, "%s, ", sig.c_str());
      saveVertVect(fp, V);
    }
    else if (V.size() > 0)
      printf("improper number of vertices for external image %s; not saving\n", sig.c_str());
  }

  fclose(fp);

  printf("saved %i images with verts [%i S + %i U = %i]\n", 
	 num_saved_verts, 
	 (int) Vert_idx_set.size(),
	 (int) NoVert_idx_set.size(),
	 (int) Vert.size());

  printf("saved %i EXTERNAL images with verts\n", 
	 (int) External_vert_vect.size());
}

//----------------------------------------------------------------------------

// utility function for reading ground-truth vertices from file

// extract next integer from string starting from index startPos
// if s[startPos] is NOT a digit, keeps searching until one is found
// if no digit found before end of string, error

bool getNextInt(string s, int startPos, int & curPos, int & nextInt)
{
  string s_nextint = "";

  for (curPos = startPos; curPos < s.length() && (isdigit(s[curPos]) || s_nextint.length() == 0); curPos++) {
    if (isdigit(s[curPos])) 
      s_nextint += s[curPos];
    //    cout << curPos << " " << s_nextint << endl;

  }

  if (s_nextint.length() > 0) {
    nextInt = atoi(s_nextint.c_str());
    return true;
  }
  else
    return false;
}

//----------------------------------------------------------------------------

// utility function for reading ground-truth coordinate pair (x, y) from file

bool getNextVert(string s, int startPos, int & curPos, int & x, int & y)
{
  // get x coord

  if (!getNextInt(s, startPos, curPos, x)) 
    return false;

  // get y coord

  if (!getNextInt(s, curPos, curPos, y)) {
    printf("loadVertMap(): x coordinate but no y! [%s]\n", s.c_str());
    exit(1);
  }

  //  printf("got %i, %i!\n", x, y); 

  return true;
}

//----------------------------------------------------------------------------

// read next datetime string + image name and reset current position

bool getNextSignature(string s, int startPos, int & curPos, string & nextSig)
{
  string s_nextsig = "";
  set<string>::iterator iter;
  std::size_t pos, endpos;

  //  printf("getNextSignature: %s\n", s.c_str());

  // find day of week -- it must be there or else error

  for (iter = dayofweek_set.begin(); iter != dayofweek_set.end(); iter++) {
    
    pos = s.find(*iter);

    // we got it

    if (pos != std::string::npos) {

      pos -= MONTH_DAY_OFFSET;

      // extract substring starting at pos and running to jpg or png ending

      endpos = s.find("jpg");
      if (endpos != std::string::npos) {
	curPos = endpos + 3;
	nextSig = s.substr(pos, curPos);
	//	printf("sig: %s\n", nextSig.c_str());
	return true;
      }
      else {
	endpos = s.find("png");
	if (endpos != std::string::npos) {
	  curPos = endpos + 3;
	  nextSig = s.substr(pos, curPos);
	  //	  printf("sig: %s\n", nextSig.c_str());
	  return true;
	}
      }
    }
  }


  return false;
}

//----------------------------------------------------------------------------

// how many "images away" from this one is the nearest image for which we have ground-truth?

// i should be the index of a nonvert image

void calculate_closest_vert_dist(int i)
{
  int j, last_dist, next_dist;

  // get NEXT vert index
  
  for (j = i + 1, next_dist = NO_DISTANCE; j < Vert.size() && (isBad(j) || !isVert(j)); j++)
    ;
  
  if (j >= 0 && j < Vert.size() && isVert(j)) 
    next_dist = j - i;
  
  // get LAST vert index 
  
  for (j = i - 1, last_dist = NO_DISTANCE; j >= 0 && (isBad(j) || !isVert(j)); j--)
    ;
  if (j >= 0 && j < Vert.size() && isVert(j)) 
    last_dist = i - j;
  
  // is forward or backward dist smaller?
  
  // at least one distance is "infinite"
  
  if (next_dist == NO_DISTANCE) 
    ClosestVert_dist[i] = last_dist;
  else if (last_dist == NO_DISTANCE) 
    ClosestVert_dist[i] = next_dist;
  
  // both distances normal
  
  else {
    if (next_dist <= last_dist) 
      ClosestVert_dist[i] = next_dist;
    else 
      ClosestVert_dist[i] = last_dist;
  }
}

//----------------------------------------------------------------------------

// a new vert has been inserted.  recalculate MINIMUM number of distances from 
// non-vert images to nearest vert image

int most_isolated_nonvert_image_idx(int new_vert_idx)
{
  int i, j, max_dist, max_dist_idx;

  // inserted vertex is now out of the game

  ClosestVert_dist[new_vert_idx] = NO_DISTANCE;

  // forward non-verts

  for (j = new_vert_idx + 1; j < Vert.size() && !isVert(j); j++)
    if (!isBad(j)) 
      calculate_closest_vert_dist(j);

  // backward non-verts

  for (j = new_vert_idx - 1; j >= 0 && !isVert(j); j--)
    if (!isBad(j)) 
      calculate_closest_vert_dist(j);

  // now find global max

  for (i = 0, max_dist = 0, max_dist_idx = NO_INDEX; i < Vert.size(); i++) 
    if (ClosestVert_dist[i] != NO_DISTANCE && ClosestVert_dist[i] > max_dist) {
      max_dist = ClosestVert_dist[i];
      max_dist_idx = i;
    }

  max_closest_vert_dist = max_dist;

  return max_dist_idx;
}


//----------------------------------------------------------------------------

// calculate distances from ALL non-vert images to nearest vert image.  
// expensive, but should be called only once, when program is initialized

int most_isolated_nonvert_image_idx()
{
  int i, j, last_dist, next_dist, max_dist, max_dist_idx;

  
  for (i = 0; i < Vert.size(); i++) {
    
    if (!isBad(i) && !isVert(i)) 
      calculate_closest_vert_dist(i);
    else
      ClosestVert_dist[i] = NO_DISTANCE;

    if (!(i % 1000)) {
      printf("%i / %i\n", i, (int) Vert.size()); 
      fflush(stdout);
    }
  }

  for (i = 0, max_dist = 0, max_dist_idx = NO_INDEX; i < Vert.size(); i++) 
    if (ClosestVert_dist[i] != NO_DISTANCE && ClosestVert_dist[i] > max_dist) {
      max_dist = ClosestVert_dist[i];
      max_dist_idx = i;
    }

  max_closest_vert_dist = max_dist;

  return max_dist_idx;
}

//----------------------------------------------------------------------------

// read all verts from string line starting at line_idx into V

void loadVertVect(string & line, int line_idx, VertVect & V)
{
  int x_val, y_val;
  Point v;

  while (getNextVert(line, line_idx, line_idx, x_val, y_val)) {

    v.x = x_val;
    v.y = y_val;
    
    //    Vert[image_idx].push_back(v);
    V.push_back(v);
  }
}

//----------------------------------------------------------------------------

bool getScallopParams(bool is_hunter, string s, ScallopParams & scparams, int & params_start_index, int & params_end_index)
{
  set<string>::iterator iter;
  std::size_t pos, endpos, len;
  string param_str;
  
  pos = s.find("Scallop");

  // we got it
  
  if (pos != std::string::npos) {

    params_start_index = pos + 8;
    int max_coord_chars = 20;   // 10
    param_str = s.substr(params_start_index, max_coord_chars); // longest possible is 1280,960,1280,960
    
    istringstream iss(param_str);
    int x1, y1, x2, y2, comma;

    // grab first x, y
    
    iss >> x1;
    iss.ignore();   // to skip comma
    iss >> y1;

    // that's all we need -- just convert from web coords to images coords

    if (is_hunter) {


      scparams.p_annotation.x = (int) rint(SCALLOP_UP*(float) x1);
      scparams.p_annotation.y = (int) rint(SCALLOP_UP*(float) y1);
      scparams.has_scale = false;
    }

    // also get x2, y2
    // if the same as x1, y1, we got nothing.  else, we have scale
    
    else {

      iss.ignore();   // to skip comma
      iss >> x2;
      iss.ignore();   // to skip comma
      iss >> y2;

      //      printf("got %i, %i and %i, %i\n", x1, y1, x2, y2);

      if (x1 != x2 && y1 != y2) {
	
	scparams.has_scale = true;

	// these values are already in image coordinate system -- no scaling necessary

	scparams.p_upper_left.x = x1;
	scparams.p_upper_left.y = y1;

	scparams.p_lower_right.x = x2;
	scparams.p_lower_right.y = y2;

      }
      else
	scparams.has_scale = false;
    }
    
    //    printf("scallop %i, %i\n", scparams.p_annotation.x, scparams.p_annotation.y);

    // line ends with "alive", "dead", or "none"

    pos = s.find("alive");
    if (pos != std::string::npos) {
      scparams.type = SCALLOP_ALIVE_TYPE;
      params_end_index = pos - 1;
    }
    else {
      pos = s.find("dead");
      if (pos != std::string::npos) {
	scparams.type = SCALLOP_DEAD_TYPE;
	params_end_index = pos - 1;
      }
      else {
	pos = s.find("none");
	if (pos != std::string::npos) {
	  scparams.type = SCALLOP_DEAD_TYPE;
	  params_end_index = pos - 1;
	}
	// none of those three words was found -- error
	else {
	  printf("category does not match alive, dead, or none -- skipping\n");
	  return false;
	}
      }
    }
    
    //    cout << param_str << endl;
    
    return true;
  }
  // else see if "Shark", "Fish", etc.
  
  return false;
}

//----------------------------------------------------------------------------

// strips jpg or ppm
 
// this works with line from csv as well as fullpathname
 
bool getScallopSignature(string s, string & sig)
{
  set<string>::iterator iter;
  std::size_t pos, endpos, len;

  pos = s.find("frame");

  // we got it
  
  if (pos != std::string::npos) {
    
    //    pos -= MONTH_DAY_OFFSET;
    
    // extract substring starting at pos and running to jpg or ppm ending
    
    endpos = s.find("ppm");   
    if (endpos != std::string::npos) {
      len = endpos - pos - 1;
      sig = s.substr(pos, len);
      return true;
    }
    else {
      endpos = s.find("jpg");
      if (endpos != std::string::npos) {
	len = endpos - pos - 1;
	sig = s.substr(pos, len);
	return true;
      }
    }
  }

  return false;
}

//----------------------------------------------------------------------------

bool filter_for_traintest_scallop(ScallopParams & scparams)
{
  int min_edge_dist = 10;

  if (!scparams.has_scale)
    return false;
  if (scparams.type != SCALLOP_ALIVE_TYPE)
    return false;
  if (scparams.p_upper_left.x < min_edge_dist || scparams.p_lower_right.x < min_edge_dist ||
      scparams.p_upper_left.x >= (SCALLOP_IMAGE_WIDTH - min_edge_dist) || scparams.p_lower_right.x >= (SCALLOP_IMAGE_WIDTH - min_edge_dist))
    return false;
  if (scparams.p_upper_left.y < min_edge_dist || scparams.p_lower_right.y < min_edge_dist || 
      scparams.p_upper_left.y >= (SCALLOP_IMAGE_HEIGHT - min_edge_dist) || scparams.p_lower_right.y >= (SCALLOP_IMAGE_HEIGHT - min_edge_dist))
    return false;
  
  return true;
}

//----------------------------------------------------------------------------

void compute_stats_traintest_scallop()
{
  vector < pair <string, ScallopParams > > scallop_traintest;
  vector < string > image_sig_vect;
  string date_str = string(UD_datetime_string());
  string image_sig;
  set < string > image_sig_set;

  printf("csv lines %i, num scallop params %i\n",
	 (int) scallop_csv_lines.size(), (int) scallop_params_vect.size());

  // filter
  
  int scallop_idx = 0;
  int w_total = 0;
  int h_total = 0;
  int w_min = 1000;
  int w_max = 0;
  int h_min = 1000;
  int h_max = 0;
  float aspect_min = 100.0;
  float aspect_max = 0.0;
  
  scallop_traintest.clear();

  for (int i = 0; i < scallop_csv_lines.size(); i++) {

    map<int, int>::iterator iter = scallop_line_idx_idx_map.find(i);
    if (iter != scallop_line_idx_idx_map.end()) {
      ScallopParams scparams = scallop_params_vect[(*iter).second];

      //    ScallopParams scparams = scallop_params_vect[i];
      if (filter_for_traintest_scallop(scparams)) {

	int w = scparams.p_lower_right.x - scparams.p_upper_left.x;
	int h = scparams.p_lower_right.y - scparams.p_upper_left.y;
	float aspect = (float) w / (float) h;
	
	w_total += w;
	h_total += h;

	if (w < w_min)
	  w_min = w;
	if (w > w_max)
	  w_max = w;

	if (h < h_min)
	  h_min = h;
	if (h > h_max)
	  h_max = h;

	if (aspect < aspect_min)
	  aspect_min = aspect;
	if (aspect > aspect_max)
	  aspect_max = aspect;

	printf("%i, %i, %i, %.2f\n",
	       scallop_idx, w, h, aspect);

	//	printf("%i, %i, %i, %i, %i\n",
	//      scallop_idx,
	//      scparams.p_upper_left.x, scparams.p_upper_left.y,
	//      scparams.p_lower_right.x, scparams.p_lower_right.y);

	
	scallop_idx++;

	
      }
    }
  }

  printf("%i scallops passed filter\n", scallop_idx);
  printf("w: min = %i, max = %i\n", w_min, w_max);
  printf("h: min = %i, max = %i\n", h_min, h_max);
  printf("aspect: min = %.2f, max = %.2f\n", aspect_min, aspect_max);
  float w_mean = (float) w_total / (float) scallop_idx;
  float h_mean = (float) h_total / (float) scallop_idx;
  float aspect_mean = w_mean / h_mean;
  printf("w mean = %.2f, h mean = %.2f, aspect mean = %.2f\n", w_mean, h_mean, aspect_mean);
}

//----------------------------------------------------------------------------

void write_traintest_scallop(string dir, float training_fraction)
{
  vector < pair <string, ScallopParams > > scallop_traintest;
  vector < string > image_sig_vect;
  string date_str = string(UD_datetime_string());
  string image_sig;
  set < string > image_sig_set;
  stringstream ss;
  string annotations_path, imagesets_path;

  ss << "mkdir " << dir << "_" << date_str;
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/Annotations";
  annotations_path = ss.str();
  ss.str("");
  ss << "mkdir " << annotations_path; 
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  ss.str("");
  ss << dir << "_" << date_str << "/ImageSets";
  imagesets_path = ss.str();
  ss.str("");
  ss << "mkdir " << imagesets_path; 
  printf("%s\n", ss.str().c_str());
  system(ss.str().c_str());

  printf("csv lines %i, num scallop params %i\n",
	 (int) scallop_csv_lines.size(), (int) scallop_params_vect.size());

  // filter
  
  int i, scallop_idx;

  scallop_traintest.clear();

  //  for (i = 0, scallop_idx = 0; i < scallop_params_vect.size(); i++) {
  for (int i = 0; i < scallop_csv_lines.size(); i++) {

    map<int, int>::iterator iter = scallop_line_idx_idx_map.find(i);
    if (iter != scallop_line_idx_idx_map.end()) {
      ScallopParams scparams = scallop_params_vect[(*iter).second];

      //    ScallopParams scparams = scallop_params_vect[i];
      if (filter_for_traintest_scallop(scparams)) {

	if (!getScallopSignature(scallop_csv_lines[i], image_sig)) {
	  printf("loadScallopMap(): problem parsing signature on line %i\n", i);
	  exit(1);
	}

	/*
	printf("line start %s\n", scallop_csv_lines[i].c_str());
	printf("line end %s\n", scallop_csv_line_endings[i].c_str());
	printf("sig %s\n", image_sig.c_str());
	printf("%i, %i, %i, %i, %i\n",
	       scallop_idx,
	       scparams.p_upper_left.x, scparams.p_upper_left.y,
	       scparams.p_lower_right.x, scparams.p_lower_right.y);
	printf("\n");
	*/
	
	scallop_idx++;

	scallop_traintest.push_back(make_pair(image_sig, scparams));
	
      }
    }
  }

  // make set of images containing filtered scallops

  for (int i = 0; i < scallop_traintest.size(); i++)     
    image_sig_set.insert(scallop_traintest[i].first);

  // move over to vector and randomize
  
  set <string>::iterator iter;

  for (iter = image_sig_set.begin(); iter != image_sig_set.end(); iter++)     
    image_sig_vect.push_back(*iter);

  random_shuffle(image_sig_vect.begin(), image_sig_vect.end());

  //  printf("scallop_traintest size %i, image_sig_set %i\n", scallop_traintest.size(), image_sig_set.size());

  // we need to do this by image, not scallop
  // make a set of image sigs, vectorize it, shuffle it, and then find the scallops that belong to each image and construct an xml for that image
  
  // remember to deal with scaling

  int num_training = (int) rint(training_fraction * (float) image_sig_vect.size());
    
  // write it

  string dir_name = string("scallop_data_") + date_str;

  int num_train_scallops = 0;
  int num_test_scallops = 0;

  FILE *ann_fp = stdout;
  char *ann_filename = (char *) malloc(sizeof(char)*512);
  
  FILE *imsets_fp = stdout;
  char *imsets_filename = (char *) malloc(sizeof(char)*512);

  FILE *imcopy_fp;
  char *imcopy_filename = (char *) malloc(sizeof(char)*512);

  ss.str("");
  ss << dir << "_" << date_str;

  sprintf(imcopy_filename, "%s/imcopy.sh", ss.str().c_str());
  imcopy_fp = fopen(imcopy_filename, "w");
  fprintf(imcopy_fp, "mkdir ./JPEGImages\n");
  
  for (int i = 0; i < image_sig_vect.size(); i++) {

    // image copy command -- so we only get the relevant images

    fprintf(imcopy_fp, "cp ~/Documents/data/scallops/images/%s.jpg JPEGImages/%06i.jpg\n",
	    image_sig_vect[i].c_str(), i);
    
    // train or test?
    
    if (i == 0) {
      sprintf(imsets_filename, "%s/train.txt", imagesets_path.c_str());
      imsets_fp = fopen(imsets_filename, "w");
    }
    else if (i == num_training) {
      fclose(imsets_fp);
      sprintf(imsets_filename, "%s/test.txt", imagesets_path.c_str());
      imsets_fp = fopen(imsets_filename, "w");
    }

    fprintf(imsets_fp, "%06i\n", i);

    // annotation data
    
    sprintf(ann_filename, "%s/%06i.xml", annotations_path.c_str(), i);
    printf("%s\n", ann_filename);
    ann_fp = fopen(ann_filename, "w"); 
    
    fprintf(ann_fp, "<annotation>\n");
    //    fprintf(ann_fp, "  <folder>scallops</folder>\n");
    fprintf(ann_fp, "  <folder>%s</folder>\n", dir_name.c_str());
    fprintf(ann_fp, "  <filename>%s.jpg</filename>\n", image_sig_vect[i].c_str());


    /*
    if (i < num_training) 
      fprintf(ann_fp, "train/%06i, %s\n", i, image_sig_vect[i].c_str());
    else 
      fprintf(ann_fp, "test/%06i, %s\n", i - num_training, image_sig_vect[i].c_str());
    */
    
    for (int j = 0; j < scallop_traintest.size(); j++) {
      string scallop_image_sig = scallop_traintest[j].first;
      if (scallop_image_sig == image_sig_vect[i])  {

	fprintf(ann_fp, "  <object>\n");
	fprintf(ann_fp, "    <name>scallop</name>\n");
	fprintf(ann_fp, "    <pose>Frontal</pose>\n");
	fprintf(ann_fp, "    <truncated>0</truncated>\n");
	fprintf(ann_fp, "    <difficult>0</difficult>\n");
	fprintf(ann_fp, "    <bndbox>\n");

	ScallopParams scparams = scallop_traintest[j].second;
	fprintf(ann_fp, "      <xmin>%i</xmin>\n", scparams.p_upper_left.x);
	fprintf(ann_fp, "      <ymin>%i</ymin>\n", scparams.p_upper_left.y);
	fprintf(ann_fp, "      <xmax>%i</xmax>\n", scparams.p_lower_right.x);
	fprintf(ann_fp, "      <ymax>%i</ymax>\n", scparams.p_lower_right.y);
	
	if (i < num_training)
	  num_train_scallops++;
	else
	  num_test_scallops++;

	fprintf(ann_fp, "    </bndbox>\n");
	fprintf(ann_fp, "  </object>\n");

      }
    }
    fprintf(ann_fp, "</annotation>\n");

    fclose(ann_fp);
  }

  fclose(imsets_fp);
  fclose(imcopy_fp);

  printf("%i training images (%i scallops), %i test images (%i scallops)\n", num_training, num_train_scallops, (int) image_sig_vect.size() - num_training, num_test_scallops); 

}
//----------------------------------------------------------------------------

void saveScallopMap()
{
  int no_params_total = 0;
  int num_with_scale = 0;
  //  FILE *fp = fopen("/home/cer/Documents/data/scallops/scale_data.csv", "w"); // stdout;
  FILE *fp = fopen("./scallop_data/scale_data.csv", "w"); // stdout;
  
  printf("# params %i\n", (int) scallop_params_vect.size());
  printf("# lines %i\n", (int) scallop_csv_lines.size());
  printf("# endings %i\n", (int) scallop_csv_line_endings.size());

  for (int i = 0; i < scallop_csv_lines.size(); i++) {
    //    fprintf(fp, "%i\n", i);
    fprintf(fp, "%s", scallop_csv_lines[i].c_str());

    map<int, int>::iterator iter = scallop_line_idx_idx_map.find(i);
    if (iter != scallop_line_idx_idx_map.end()) {
      //printf("params\n");
      ScallopParams scparams = scallop_params_vect[(*iter).second];
      if (scparams.has_scale) {
	fprintf(fp, "%i,%i,%i,%i,",
		scparams.p_upper_left.x, scparams.p_upper_left.y,
		scparams.p_lower_right.x, scparams.p_lower_right.y);
	num_with_scale++;
      }
      else
	fprintf(fp, "-1,-1,-1,-1,");
      fprintf(fp, "%s\n", scallop_csv_line_endings[i].c_str());
    }
    else {
      //printf("NO params\n");
      //      fprintf(fp, "-1,-1,-1,-1,");
      fprintf(fp, "\n");
      no_params_total++;
    }
    //    fprintf(fp, "ending %s\n", scallop_csv_line_endings[i].c_str());
  }

  fclose(fp);
  
  printf("%i / %i\n", no_params_total, (int) scallop_csv_lines.size());
  printf("%i scallops with scale saved\n", num_with_scale);
}

//----------------------------------------------------------------------------

// THIS IS A FISH!!! problem at 1389 ./scallop_data/images/frame004911_1436666595_247114.jpg
// i changed this manually to Fish in hunter_data.csv and scale_data.csv
// we will take only live scallops that are completely inside the image -- every corner >= 10 pixel
// from image edges

int loadScallopMap(bool is_hunter)
{
  ifstream inStream;
  string line;
  int total_lines = 0;
  int total_images = 0;
  int total_scallops = 0;
  string line_precoords, line_postcoords;
  int params_start_index, params_end_index;
  int line_num = 0;
  int scallop_idx;
  int num_scallops_with_scale = 0;
  
  string image_sig;
  string imname;

  // read file

  if (is_hunter) 
    //    inStream.open("/home/cer/Documents/data/scallops/hunter_data.csv");
    inStream.open("./scallop_data/hunter_data.csv");
  else
    //    inStream.open("/home/cer/Documents/data/scallops/scale_data.csv");
    inStream.open("./scallop_data/scale_data.csv");

  while (getline(inStream, line)) {

    line_num++;
    
    // skip comments

    if (line[0] == COMMENT_CHAR) {
      if (is_hunter) {
	scallop_csv_lines.push_back(line);
	scallop_csv_line_endings.push_back("");
      }
      continue;
    }

    // get image signature

    if (!getScallopSignature(line, image_sig)) {
      printf("loadScallopMap(): problem parsing signature on line %i\n", total_lines);
      exit(1);
    }

    //    printf("nextsig = %s\n", image_sig.c_str());

    total_lines++;

    // see if image is there
    // note that same image should be on different lines if it contains multiple scallops

    //    imname = string("/home/cer/Documents/data/scallops/images/" + image_sig + ".jpg");
    imname = string("./scallop_data/images/" + image_sig + ".jpg");

    //Mat im = imread(imname.c_str());
    //if (!im.data)
    //  continue;
    FILE *fp = fopen(imname.c_str(), "r");
    if (!fp) {
      if (is_hunter) {
	scallop_csv_lines.push_back(line);
	scallop_csv_line_endings.push_back("");
      }
      continue;
    }
    fclose(fp);
    
    total_images++;

    // see if scallop is there

    ScallopParams scparams;
    
    if (!getScallopParams(is_hunter, line, scparams, params_start_index, params_end_index)) {
      if (is_hunter) {
	scallop_csv_lines.push_back(line);      
	scallop_csv_line_endings.push_back("");
      }
      continue;
    }

    if (is_hunter) {
      scallop_csv_lines.push_back(line.substr(0, params_start_index));
      // sc params go in between with trailing comma: "x_ul, y_ul, x_lr, y_lr,"
      scallop_csv_line_endings.push_back(line.substr(params_end_index + 1, line.length() - params_end_index));
      //      printf("line %s\n", scallop_csv_lines[scallop_csv_lines.size()-1].c_str());
      //      printf("ending %s\n", scallop_csv_line_endings[scallop_csv_line_endings.size()-1].c_str());
    
      scallop_idx_line_idx_map[scallop_params_vect.size()] = scallop_csv_lines.size()-1;
      scallop_line_idx_idx_map[scallop_csv_lines.size()-1] = scallop_params_vect.size();

      //      printf("line %i\n", line_num-1);
      //      printf("idx -> line_idx = %i -> %i\n", scallop_params_vect.size(), scallop_csv_lines.size()-1);
      //      printf("line_idx -> idx = %i -> %i\n", scallop_csv_lines.size()-1, scallop_params_vect.size());
    
      scallop_idx_signature_map[scallop_params_vect.size()] = image_sig;
      scallop_params_vect.push_back(scparams);
      scallop_Fullpathname_set.insert(imname);
    
      //    printf("%s\n", image_sig.c_str());
      //    printf("%s\n", line.c_str());

      total_scallops++;
      //    cout << line << endl;
    }
    else {
      // put scparams upper_left and lower_right into scallop_params_vect at correct location

      if (scparams.has_scale) {

	num_scallops_with_scale++;
	
	scallop_idx = scallop_line_idx_idx_map[line_num - 1];
	//	printf("line %i, scallop %i (%i)\n", line_num - 1, scallop_idx, (int) scallop_params_vect.size());

	scallop_params_vect[scallop_idx].p_upper_left = scparams.p_upper_left;
	scallop_params_vect[scallop_idx].p_lower_right = scparams.p_lower_right;
	scallop_params_vect[scallop_idx].has_scale = true;
      }
      else {
	printf("problem at %i %s ? \n", scallop_line_idx_idx_map[line_num - 1], imname.c_str());
      }
    }
  }

  inStream.close();

  // convert set to vect

  if (is_hunter) {
    
    set<string>::iterator iter;
    
    for (iter = scallop_Fullpathname_set.begin(); iter != scallop_Fullpathname_set.end(); iter++) {
      
      imname = *iter;
      scallop_Fullpathname_vect.push_back(imname);
    }
    
    printf("total lines = %i\n", total_lines);
    printf("total images found = %i\n", total_images);
    printf("total scallops found = %i\n", total_scallops);
    printf("total images containing scallops = %i\n", (int) scallop_Fullpathname_set.size());
  }
  else {
    printf("%i scallops with scale loaded\n", num_scallops_with_scale);
  }
}

//----------------------------------------------------------------------------

// get all ground-truth trail vertices from **MAP** file

// dataset should already be loaded

int loadVertMap()
{
  ifstream inStream;
  string line;
  int i, line_idx, image_idx, x_val, y_val;
  Point v;
  string image_sig;
  map<string, int>::iterator iter;
  int total = 0;

  // initialize NoVert_idx_set to all images

  for (i = 0; i < Vert.size(); i++)
    NoVert_idx_set.insert(i);

  // read file

  inStream.open("vertmap.txt");

  while (getline(inStream, line)) {

    // skip comments

    if (line[0] == COMMENT_CHAR)
      continue;

    // get image signature

    if (!getNextSignature(line, 0, line_idx, image_sig)) {
      printf("loadVertMap(): problem parsing signature on line %i\n", total);
      exit(1);
    }

    //    printf("nextsig = %s\n", image_sig.c_str());

    iter = Signature_idx_map.find(image_sig);

    // if it IS in our current dataset, add vert info

    if (iter != Signature_idx_map.end()) {
      image_idx = (*iter).second;
      Vert_idx_set.insert(image_idx);
      //      printf("idx %i\n", (*iter).second);

      set<int>::iterator nv_iter = NoVert_idx_set.find(image_idx);
      // if it is actually in the set, erase it
      if (nv_iter != NoVert_idx_set.end())
	NoVert_idx_set.erase(nv_iter);

      loadVertVect(line, line_idx, Vert[image_idx]);
    }

    // it's NOT -- store it somewhere else so it's saved properly

    else {

      VertVect V;

      loadVertVect(line, line_idx, V);

      External_vert_vect.push_back(make_pair(image_sig, V));

      //      printf("VERT signature not in dataset: %s\n", image_sig.c_str());
    }

    total++;
  }
 
  inStream.close();

  float coverage_frac = (float) Vert_idx_set.size() / (float) Vert.size();
  printf("vert coverage = %.3f\n", coverage_frac);
  if (coverage_frac >= MOST_ISOLATED_COVERAGE_FRACTION)
    next_nonvert_idx = most_isolated_nonvert_image_idx();

  //  filter_for_traintest();

  return total;
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

//! of 3-vectors

vector <float> old_cross_product(vector <float> v1, vector <float> v2)
{
  vector <float> result;

  result.clear();

  result.push_back(v1[1] * v2[2] - v1[2] * v2[1]);
  result.push_back(v1[2] * v2[0] - v1[0] * v2[2]);
  result.push_back(v1[0] * v2[1] - v1[1] * v2[0]);

  return result;
}

//----------------------------------------------------------------------------

//! homogeneous representation of a line through two points

vector <float> old_make_line(Point p1, Point p2)
{
  vector <float> hp1, hp2;

  hp1.clear();
  hp2.clear();

  hp1.push_back(p1.x);
  hp1.push_back(p1.y);
  hp1.push_back(1);

  hp2.push_back(p2.x);
  hp2.push_back(p2.y);
  hp2.push_back(1);

  return old_cross_product(hp1, hp2);
}

//----------------------------------------------------------------------------

float point_line_distance(Point q, Point p1, Point p2)
{
  vector <float> pline;
  float dist;
  
  pline = old_make_line(p1, p2);
  float norm = sqrt(pline[0]*pline[0] + pline[1]*pline[1]);
  pline[0] /= norm;
  pline[1] /= norm;
  pline[2] /= norm;
  
  dist = (float) q.x * pline[0] + (float) q.y * pline[1] + pline[2];
  //dist = (float) q.x * (pline[0] / pline[2]) + (float) q.y * (pline[1] / pline[2]) + 1.0; // pline[2];

  return dist;
}

//----------------------------------------------------------------------------

#define EPS 0.000001

//! make a line p from two points and another line p' from another two points
//! and find intersection of p and p' iff it is within both segments

//! return true if there is a good intersection, false otherwise

bool line_segment_intersection(Point p1, Point p2, Point p1_prime, Point p2_prime, Point & p_inter)
{
  vector <float> pline, pprimeline, inter;
  Point p_diff, p_prime_diff;
  double p_len, p_prime_len;
  double u, u_prime;

  pline = old_make_line(p1, p2);
  pprimeline = old_make_line(p1_prime, p2_prime);

  inter = old_cross_product(pline, pprimeline);

  if (inter[2]) {

    // infinite lines do intersect

    p_inter.x = inter[0] / inter[2];
    p_inter.y = inter[1] / inter[2];

    // now see if it's in segment 1 

    p_diff.x = p2.x - p1.x;
    p_diff.y = p2.y - p1.y;
    p_len = sqrt(p_diff.x * p_diff.x + p_diff.y * p_diff.y);

    if (p_len) {
      
      // these two lines are unnecessary, which caused the intersection pts errors
      //p_diff.x /= p_len; 
      //p_diff.y /= p_len;
      
      if (fabs(p_diff.x) > EPS)
	u = (p_inter.x - p1.x) / p_diff.x;
      else
	u = (p_inter.y - p1.y) / p_diff.y;
    }
    else
      u = 0.0;

    if (u < 0.0 || u > 1.0) {   // >=
      //    printf("fail A: %lf\n", u);
      return false;
    }

    // how about segment 2

    p_prime_diff.x = p2_prime.x - p1_prime.x;
    p_prime_diff.y = p2_prime.y - p1_prime.y;
    p_prime_len = sqrt(p_prime_diff.x * p_prime_diff.x + p_prime_diff.y * p_prime_diff.y);

    if (p_prime_len) {
      
      // these two lines are unnecessary, which caused the intersection pts errors
      //p_prime_diff.x /= p_prime_len;
      //p_prime_diff.y /= p_prime_len;
      
      if (fabs(p_prime_diff.x) > EPS)
	u_prime = (p_inter.x - p1_prime.x) / p_prime_diff.x;
      else
	u_prime = (p_inter.y - p1_prime.y) / p_prime_diff.y;
    }
    else
      u_prime = 0.0;

    if (u_prime < 0.0 || u_prime > 1.0) {    // >= 1
      //      printf("fail B: %lf\n", u_prime);
      return false;
    }
    else {
      /*printf("p1 = (%.2f, %.2f), p2 = (%.2f, %.2f), p1' = (%.2f, %.2f), p2' = (%.2f, %.2f), p_inter = (%.2f, %.2f) \n", p1.x, p1.y, p2.x, p2.y, p1_prime.x, p1_prime.y, p2_prime.x, p2_prime.y, p_inter.x, p_inter.y);
      printf("u = %f, ", u);
      printf("u_prime = %f \n", u_prime);*/

      return true;
    }
  }
  else {
    //    printf("fail C\n");
    return false;
  }
}

//! make a line p from two points and another line p' from another two points
//! and find intersection of p and p' iff it is within p' (i.e., treat p as RAY)

//! p1 is origin of ray.  p2 - p1 defines direction of ray, but it is not
//! assumed to be a unit vector.  in fact, if len(p1, p2) < delta_thresh,
//! ignore entirely

//! return true if there is a good intersection, false otherwise

bool ray_line_segment_intersection(Point p1, Point p2, Point p1_prime, Point p2_prime, double delta_thresh, Point & p_inter)
{
  vector <float> pline, pprimeline, inter;
  Point p_diff, p_prime_diff;
  double p_len, p_prime_len;
  double u, u_prime;

  pline = old_make_line(p1, p2);
  pprimeline = old_make_line(p1_prime, p2_prime);

  inter = old_cross_product(pline, pprimeline);

  if (inter[2]) {

    // infinite lines do intersect

    p_inter.x = inter[0] / inter[2];
    p_inter.y = inter[1] / inter[2];

    //    printf("p_inter %i %i\n", p_inter.x, p_inter.y);
    
    // now see if it is on ray -- i.e. on line, but "in front of" p1 rather than "behind" it

    p_diff.x = p2.x - p1.x;
    p_diff.y = p2.y - p1.y;
    p_len = sqrt(p_diff.x * p_diff.x + p_diff.y * p_diff.y);

    // make sure p1 and p2 are far enough apart
    
    if (p_len < delta_thresh)
      return false;
    
    if (p_len) {
      
      // these two lines are unnecessary, which caused the intersection pts errors
      //p_diff.x /= p_len; 
      //p_diff.y /= p_len;
      
      if (fabs(p_diff.x) > EPS)
	u = (double) (p_inter.x - p1.x) / (double) p_diff.x;
      else
	u = (double) (p_inter.y - p1.y) / (double) p_diff.y;
    }
    else
      u = 0.0;

    //    printf("u %lf\n", u);
    
    if (u <= 0.0) {  // only one inequality for a ray, vs. two for line segment -> "|| u > 1.0") {   // >=
      //    printf("fail A: %lf\n", u);
      return false;
    }

    //    printf("passed ray\n");
    
    // now is it in line segment between p1_prime and p2_prime??

    p_prime_diff.x = p2_prime.x - p1_prime.x;
    p_prime_diff.y = p2_prime.y - p1_prime.y;
    p_prime_len = sqrt(p_prime_diff.x * p_prime_diff.x + p_prime_diff.y * p_prime_diff.y);

    if (p_prime_len) {
      
      // these two lines are unnecessary, which caused the intersection pts errors
      //p_prime_diff.x /= p_prime_len;
      //p_prime_diff.y /= p_prime_len;
      
      if (fabs(p_prime_diff.x) > EPS)
	u_prime = (double) (p_inter.x - p1_prime.x) / (double) p_prime_diff.x;
      else
	u_prime = (double) (p_inter.y - p1_prime.y) / (double) p_prime_diff.y;
    }
    else
      u_prime = 0.0;

    //    printf("p_prime_diff %i %i\n", p_prime_diff.x, p_prime_diff.y);
    //    printf("p_prime_len %lf\n", p_prime_len);
    //    printf("u_prime %lf\n", u_prime);
    
    if (u_prime < 0.0 || u_prime > 1.0) {    // >= 1
      //      printf("fail B: %lf\n", u_prime);
      return false;
    }
    else {
      /*printf("p1 = (%.2f, %.2f), p2 = (%.2f, %.2f), p1' = (%.2f, %.2f), p2' = (%.2f, %.2f), p_inter = (%.2f, %.2f) \n", p1.x, p1.y, p2.x, p2.y, p1_prime.x, p1_prime.y, p2_prime.x, p2_prime.y, p_inter.x, p_inter.y);
      printf("u = %f, ", u);
      printf("u_prime = %f \n", u_prime);*/

      return true;
    }
  }
  else {
    //    printf("fail C\n");
    return false;
  }
}

//----------------------------------------------------------------------------

bool ray_image_boundaries_intersection(Point p1, Point p2, double delta_thresh, Point & p_inter)
{
  bool result;
  
  result = ray_line_segment_intersection(p1, p2, p_topleft, p_topright, RAY_DELTA_THRESH, p_inter);
  if (!result) { 
    result = ray_line_segment_intersection(p1, p2, p_topleft, p_bottomleft, RAY_DELTA_THRESH, p_inter);
    if (!result) {
      result = ray_line_segment_intersection(p1, p2, p_topright, p_bottomright, RAY_DELTA_THRESH, p_inter);
      if (!result)
	result = ray_line_segment_intersection(p1, p2, p_bottomleft, p_bottomright, RAY_DELTA_THRESH, p_inter);
    }
  }

  return result;
}

//----------------------------------------------------------------------------

int count_line_segment_poly_intersections_save(Point p1, Point p2, vector <Point> & poly, vector <Point> & intersection_pts, bool save_intersection_pts)
{
  int i, total;
  Point p_inter;

  for (i = 0, total = 0; i < poly.size(); i++) 
    if (line_segment_intersection(p1, p2, poly[i], poly[(i + 1) % poly.size()], p_inter)) {
      total++;
   
      if (save_intersection_pts) {
        //p_inter.x = -p_inter.x;
        intersection_pts.push_back(p_inter);

        /*p1.x = p1.x;
        intersection_pts.push_back(p1);

        p2.x = p2.x;
        intersection_pts.push_back(p2);

        intersection_pts.push_back(poly[i]);
        intersection_pts.push_back(poly[(i + 1) % poly.size()]);*/
      } 

      //printf("intersection: (%f, %f) \n", p_inter.x, p_inter.y);
    }

  return total;
}

//----------------------------------------------------------------------------

//! read in a polygon from an xml file written by LabelMe.
//! if file doesn't exist or anything is wrong, return empty polygon

vector <Point> load_xml_polygon(string xml_filename, float scale_factor)
{
  TiXmlDocument doc;
  TiXmlNode *node;
  vector <Point> poly_pts;
  Point p;
  int j;

  //  printf("trying to load %s\n", xml_filename.c_str());

  poly_pts.clear();

  // is there such a file?  if not, just return empty polygon

  if (doc.LoadFile(xml_filename.c_str())) {
 
    for (j = 0, node = doc.FirstChild("annotation")->FirstChild("object")->FirstChild("polygon")->FirstChild("pt"); node; node = node->NextSibling(), j++) {

      p.x = scale_factor * atof(node->FirstChildElement("x")->GetText());
      p.y = scale_factor * atof(node->FirstChildElement("y")->GetText());

      //      printf("%i: %.0f %.0f\n", j, p.x, p.y);
      //      fflush(stdout);

      // any point too close to edge of image -> poly is clipped -> treat it as empty

      /*
      if (p.x < IMAGE_FRAME_WIDTH || p.x >= IMAGE_WIDTH - IMAGE_FRAME_WIDTH) {
	poly_pts.clear();
	return poly_pts;
      }
      */

      // all is well--add this point to poly

      poly_pts.push_back(p);

    }
  }

  return poly_pts;
}

//----------------------------------------------------------------------------

Point zmax_xform(Point p)
{
  Point q;

  q.x = p.y;
  q.y = 640 - p.x;

  return q;
}

//----------------------------------------------------------------------------

string zmax_imname_to_xmlname(string imname, string & signature)
{
  string snum, spre, s;
  int len;
  std::size_t pos;

  len = imname.length();

  snum = imname.substr(len - 10, 6);

  pos = imname.rfind("omni");
  spre = imname.substr(0, pos);

  //  printf("%s\n", spre.c_str());

  s = spre + string("omni_gt_center_xmls/center_") + snum + string(".xml");

  if (imname.find("field") != string::npos)
    signature = string("Aug_28_2009_Fri_12_30_57_PM/omni_images/") + snum + string(".jpg");
  else if (imname.find("mixed") != string::npos)
    signature = string("Aug_28_2009_Fri_12_36_14_PM/omni_images/") + snum + string(".jpg");
  else
    signature = string("Aug_28_2009_Fri_12_46_56_PM/omni_images/") + snum + string(".jpg");

  return s;
}

//----------------------------------------------------------------------------

void zmax()
{
  int i, j, k;
  vector <string> zmax_image_filename;
  string xml_filename; //  =   string("/warthog_logs/zmax4_Aug_28_2009/field/omni_gt_center_xmls/center_000013.xml");
  string image_filename; //  = string("/warthog_logs/zmax4_Aug_28_2009/field/omni_images/000013.jpg");

  vector <Point> poly_pts;
  Mat M;
  string signature;

  add_images("/warthog_logs/zmax4_Aug_28_2009/field/omni_images/", zmax_image_filename);
  add_images("/warthog_logs/zmax4_Aug_28_2009/mixed/omni_images/", zmax_image_filename);
  add_images("/warthog_logs/zmax4_Aug_28_2009/forest/omni_images/", zmax_image_filename);

  for (j = 0; j < zmax_image_filename.size(); j++) {

    vector <Point> inter_pts, filtered_inter_pts;

    //    printf("%i: %s\n", j, zmax_image_filename[j].c_str());


    //  Mat M = Mat::zeros(480, 640, CV_8UC3);
    //  Mat M = Mat::zeros(CANONICAL_IMAGE_HEIGHT, CANONICAL_IMAGE_WIDTH, CV_8UC3);
    
    M = smart_imread(zmax_image_filename[j]);
    
    xml_filename = zmax_imname_to_xmlname(zmax_image_filename[j], signature);
    //    printf("%s\n", xml_filename.c_str());

    poly_pts = load_xml_polygon(xml_filename);
    for (i = 0; i < poly_pts.size(); i++) 
      poly_pts[i] = zmax_xform(poly_pts[i]);

    for (i = 0; i < poly_pts.size() - 1; i++) {
      //    printf("%i: (%i, %i)\n", i, poly_pts[i].x, poly_pts[i].y);
      
      //      line(M, zmax_xform(Point(poly_pts[i].x, poly_pts[i].y)), zmax_xform(Point(poly_pts[i+1].x, poly_pts[i+1].y)), Scalar(255, 255, 255), 1);
      line(M, poly_pts[i], poly_pts[i+1], Scalar(255, 255, 255), 1);
      //    line(M, Point(poly_pts[i].x, poly_pts[i].y), Point(poly_pts[i+1].x, poly_pts[i+1].y), Scalar(255, 255, 255), 1);
      
    }
    //    line(M, zmax_xform(Point(poly_pts[poly_pts.size()-1].x, poly_pts[poly_pts.size()-1].y)), zmax_xform(Point(poly_pts[0].x, poly_pts[0].y)), Scalar(255, 255, 255), 1);
    line(M, poly_pts[poly_pts.size()-1], poly_pts[0], Scalar(255, 255, 255), 1);

    // horizontal lines for trail edge rows

    filtered_inter_pts.clear();

    for (i = 0; i < trailEdgeRow.size(); i++) {

      inter_pts.clear();  
      vector <float> inter_x;

      line(M, Point(0, trailEdgeRow[i]), Point(M.cols - 1, trailEdgeRow[i]), Scalar(0, 128, 128), 1);

      int num = count_line_segment_poly_intersections_save(Point(0, trailEdgeRow[i]), 
							   Point(M.cols - 1, trailEdgeRow[i]), 
							   poly_pts, 
							   inter_pts, true);

      for (k = 0; k < inter_pts.size(); k++)
	inter_x.push_back(inter_pts[k].x);
      sort(inter_x.begin(), inter_x.end());


      filtered_inter_pts.push_back(Point(inter_x[0], trailEdgeRow[i]));
      filtered_inter_pts.push_back(Point(inter_x[inter_x.size()-1], trailEdgeRow[i]));

      //      printf(">>>>>>>>>>>>>>>>>>> %i: num %i\n", trailEdgeRow[i], num);
	
    }

   
    for (i = 0; i < filtered_inter_pts.size(); i++) {
      //   printf("%i: (%i, %i)\n", i, filtered_inter_pts[i].x, filtered_inter_pts[i].y);
      circle(M, filtered_inter_pts[i], 8, Scalar(0, 255, 255), 1, 8, 0);
    }
   

    printf("%s, ", signature.c_str());

    for (i = 0; i < filtered_inter_pts.size()-1; i++) 
      printf("(%i, %i), ", filtered_inter_pts[i].x, filtered_inter_pts[i].y);
    printf("(%i, %i)\n", filtered_inter_pts[i].x, filtered_inter_pts[i].y);

    // display

    imshow("zmax", M);
    char c = waitKey(5);
    
  }

  exit(1);

}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

int main( int argc, const char** argv )
{
  s_imagedirs = string("imagedirs.txt");

  // splash

  printf("hello, map_trail!\n");

  // initialize for LINEAR trail approximation (quadrilateral)

  dayofweek_set.insert("Mon");
  dayofweek_set.insert("Tue");
  dayofweek_set.insert("Wed");
  dayofweek_set.insert("Thu");
  dayofweek_set.insert("Fri");
  dayofweek_set.insert("Sat");
  dayofweek_set.insert("Sun");

  trailEdgeRow.push_back(IMAGE_ROW_FAR);
  trailEdgeRow.push_back(IMAGE_ROW_NEAR);

  p_topleft = Point(0, 0);
  p_topright = Point(CANONICAL_IMAGE_WIDTH - 1, 0);
  p_bottomleft = Point(0, CANONICAL_IMAGE_HEIGHT - 1);
  p_bottomright = Point(CANONICAL_IMAGE_WIDTH - 1, CANONICAL_IMAGE_HEIGHT - 1);
  
  // command-line options?

  if (argc == 2) {
    if (!strcmp(argv[1], "tree") || !strcmp(argv[1], "-tree")) {
      object_input_mode = TREE_MODE;
    }
    else if (!strcmp(argv[1], "trail") || !strcmp(argv[1], "-trail")) {
      object_input_mode = TRAIL_MODE;
    }
    else if (!strcmp(argv[1], "scallop") || !strcmp(argv[1], "-scallop")) {
      object_input_mode = SCALLOP_MODE;
    }
    else if (!strcmp(argv[1], "help")) {
      onKeyPress('=', true);
      exit(1);
    }
    else if (!strcmp(argv[1], "zmax")) {
      zmax();
      exit(1);
    }
    else if (!strcmp(argv[1], "traj")) {
      compute_trajectory();
      exit(1);
    }
    /*
    else if (!strcmp(argv[1], "dtrain")) {
      full_annotate_dynamics(true);
      //     annotate_dynamics(true);
      exit(1);
    }
    */
    else if (!strcmp(argv[1], "dtest")) {
      full_annotate_dynamics(false);
      //annotate_dynamics(false);
      exit(1);
    }
    else
      s_imagedirs = string(argv[1]);
  }

  // proceed

  add_all_images_from_file(s_imagedirs);

  // create initial, ordered indices 

  int i;

  for (i = 0; i < Fullpathname_vect.size(); i++)      
    Random_idx.push_back(i);
  printf("%i total images\n", (int) Random_idx.size());

  // shuffle indices 

  struct timeval tp;
  
  gettimeofday(&tp, NULL); 
  //  srand48(tp.tv_sec);
  srand(tp.tv_sec);
  random_shuffle(Random_idx.begin(), Random_idx.end());

  Nonrandom_idx.resize(Random_idx.size());
  for (i = 0; i < Random_idx.size(); i++)
    Nonrandom_idx[Random_idx[i]] = i;

  int total;

  // trail stuff -- commented out to work on trees
  
  if (object_input_mode == TRAIL_MODE || object_input_mode == TREE_MODE) {

    total = loadBadMap();
    printf("bad: %i current, %i external = %i total\n", (int) Bad_idx_set.size(), (int) External_bad_sig_vect.size(), total);
    
    Vert.resize(Random_idx.size());
    ClosestVert_dist.resize(Random_idx.size());   
    
    total = loadVertMap();
    num_saved_verts = Vert_idx_set.size();
    printf("vert: %i current, %i external = %i total\n", (int) Vert_idx_set.size(), (int) External_vert_vect.size(), total);
  }

  else if (object_input_mode == SCALLOP_MODE) {
    total = loadScallopMap(true);
    loadScallopMap(false);
  }

  if (object_input_mode == TREE_MODE) {
    set_current_index(*Vert_idx_set.begin());
    do_verts = true;
  }
  else 
    set_current_index(ZERO_INDEX);

  // display

  char c;

  do {

    if (object_input_mode == SCALLOP_MODE) {
      
      current_imname = scallop_Fullpathname_vect[current_index];
      
      current_im = imread(current_imname);
      draw_im = current_im.clone();

      scallop_draw_overlay();
    }
    else {
      
      // load image

      current_imname = Fullpathname_vect[current_index];
      
      current_im = smart_imread(current_imname);
      draw_im = current_im.clone();
      
      // show image 
      
      if (object_input_mode == TRAIL_MODE) {
	trail_draw_overlay();
	trail_draw_other_windows();
      }
      else if (object_input_mode == TREE_MODE)
	tree_draw_overlay();
    }
    
    imshow("trailGT", draw_im);

    if (!callbacks_set) {
      if (object_input_mode == TRAIL_MODE)
	setMouseCallback("trailGT", trail_onMouse);
      else if (object_input_mode == TREE_MODE)
	setMouseCallback("trailGT", tree_onMouse);
      else if (object_input_mode == SCALLOP_MODE)
	setMouseCallback("trailGT", scallop_onMouse);
      callbacks_set = true;
    }

    c = waitKey(0);

    onKeyPress(c);

  } while (c != (int) 'q');

  return 0;
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
