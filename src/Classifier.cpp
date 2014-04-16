#include "Classifier.h"

using namespace std;
using namespace pcl;
using namespace cv;

Mat imread_depth(const char* fname, bool binary) {
	char* ext = PathFindExtension(fname);
	const char char_dep[] = ".dep";
	const char char_png[] = ".png";
	Mat out;
	if(_strnicmp(ext,char_dep,strlen(char_dep))==0) {
		FILE *fp;
		if(binary)
			fp = fopen(fname,"rb");
		else
			fp = fopen(fname,"r");
		int width = 640, height = 480; //If messed up, just assume
		if(binary) {
			fread(&width,sizeof(int),1,fp);
			fread(&height,sizeof(int),1,fp);
			out = Mat(height,width,CV_32S);
			int *p = (int*)out.data;
			fread(p,sizeof(int),width*height,fp);
		} else {
			//fscanf(fp,"%i,%i,",&width,&height);
			out = Mat(height,width,CV_32S);
			int *p = (int*)out.data, *end = ((int*)out.data) + out.rows*out.cols;
			while(p != end) {
				fscanf(fp,"%i",p);
				p++;
			}
		}
		fclose(fp);
	} else if(_strnicmp(ext,char_png,strlen(char_png))==0) {
		out = cvLoadImage(fname,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
		out.convertTo(out, CV_32S);
		int* pi = (int*)out.data;
		for (int y=0; y < out.rows; y++) {
			for (int x=0; x < out.cols; x++) {
				*pi = Round(*pi * 0.2f);
				pi++;
			}
		}
	} else {
		throw std::exception("Filetype not supported");
	}
	return out;
}

Mat imread_float(const char* fname, bool binary) {
	char* ext = PathFindExtension(fname);
	const char char_dep[] = ".flt";
	Mat out;
	if(_strnicmp(ext,char_dep,strlen(char_dep))==0) {
		FILE *fp;
		if(binary)
			fp = fopen(fname,"rb");
		else
			fp = fopen(fname,"r");
		int width = 640, height = 480; //If messed up, just assume
		if(binary) {
			fread(&width,sizeof(int),1,fp);
			fread(&height,sizeof(int),1,fp);
			out = Mat(height,width,CV_32F);
			float *p = (float*)out.data;
			fread(p,sizeof(float),width*height,fp);
		} else {
			//fscanf(fp,"%i,%i,",&width,&height);
			out = Mat(height,width,CV_32F);
			float *p = (float*)out.data, *end = ((float*)out.data) + out.rows*out.cols;
			while(p != end) {
				fscanf(fp,"%f",p);
				p++;
			}
		}
		fclose(fp);
	} else {
		throw std::exception("Filetype not supported");
	}
	return out;
}

void imwrite_depth(const char* fname, Mat &img, bool binary) {
	char* ext = PathFindExtension(fname);
	const char char_dep[] = ".dep";
	Mat out;
	if(_strnicmp(ext,char_dep,strlen(char_dep))==0) {
		FILE *fp;
		if(binary)
			fp = fopen(fname,"wb");
		else
			fp = fopen(fname,"w");
		int width = img.cols, height = img.rows; //If messed up, just assume
		if(binary) {
			fwrite(&width,sizeof(int),1,fp);
			fwrite(&height,sizeof(int),1,fp);
			int *p = (int*)img.data;
			fwrite(p,sizeof(int),width*height,fp);
		} else {
			//fscanf(fp,"%i,%i,",&width,&height);
			int *p = (int*)img.data, *end = ((int*)img.data) + width*height;
			while(p != end) {
				fscanf(fp,"%f",p);
				p++;
			}
		}
		fclose(fp);
	} else {
		throw std::exception("Filetype not supported");
	}
}

void imwrite_float(const char* fname, Mat &img, bool binary) {
	char* ext = PathFindExtension(fname);
	const char char_dep[] = ".flt";
	Mat out;
	if(_strnicmp(ext,char_dep,strlen(char_dep))==0) {
		FILE *fp;
		if(binary)
			fp = fopen(fname,"wb");
		else
			fp = fopen(fname,"w");
		int width = img.cols, height = img.rows; //If messed up, just assume
		if(binary) {
			fwrite(&width,sizeof(int),1,fp);
			fwrite(&height,sizeof(int),1,fp);
			float *p = (float*)img.data;
			fwrite(p,sizeof(float),width*height,fp);
		} else {
			//fscanf(fp,"%i,%i,",&width,&height);
			float *p = (float*)img.data, *end = ((float*)img.data) + width*height;
			while(p != end) {
				fscanf(fp,"%f",p);
				p++;
			}
		}
		fclose(fp);
	} else {
		throw std::exception("Filetype not supported");
	}
}

void GetMatFromCloud(const PointCloudBgr &cloud, Mat &img) {
	img = Mat(cloud.height,cloud.width,CV_8UC3);
	Mat_<Vec3b>::iterator pI = img.begin<Vec3b>();
	PointCloudBgr::const_iterator pC = cloud.begin();
	while(pC != cloud.end()) {
		(*pI)[0] = pC->b;
		(*pI)[1] = pC->g;
		(*pI)[2] = pC->r;
		++pI; ++pC;
	}
}

void GetMatFromCloud(const PointCloudInt &cloud, Mat &img) {
	img = Mat(cloud.height,cloud.width,CV_32S);
	Mat_<int>::iterator pI = img.begin<int>();
	PointCloudInt::const_iterator pC = cloud.begin();
	while(pC != cloud.end()) {
		*pI = pC->intensity;
		++pI; ++pC;
	}
}

void Classifier::CalculateSIFTFeatures(const PointCloudBgr &cloud, Mat &descriptors) {
	Mat gImg, desc;
	ConvertCloudtoGrayMat(cloud,gImg);
	vector<KeyPoint> kp;
	featureDetector->detect(gImg,kp);
	descriptorExtractor->compute(gImg,kp,desc);
	//keypoints.push_back(kp);
	descriptors.push_back(desc);
}

void Classifier::CalculateBOWFeatures(const PointCloudBgr &cloud, Mat &descriptors) {
	Mat gImg, desc;
	ConvertCloudtoGrayMat(cloud,gImg);
	vector<KeyPoint> kp;
	featureDetector->detect(gImg,kp);
	bowDescriptorExtractor->compute(gImg,kp,desc);
	//keypoints.push_back(kp);
	descriptors.push_back(desc);
}

void Classifier::build_vocab() {
	cout << "Building vocabulary" << endl;
	// Mat to hold SURF descriptors for all templates
	// For each template, extract SURF descriptors and pool them into vocab_descriptors
	cout << "Building SURF Descriptors..." << endl;
	Mat descriptors;
	PointCloudBgr cloud;

	for(boost::filesystem::directory_iterator i(direc), end_iter; i != end_iter; i++) {
		string filename = string(direc) + i->path().filename().string();
		pcl::io::loadPCDFile(filename,cloud);
		CalculateSIFTFeatures(cloud,descriptors);
	}

	// Add the descriptors to the BOW trainer to cluster
	cout << "Training BOW..." << endl;
	bowtrainer->add(descriptors);
	// cluster the SURF descriptors
	cout << "Clustering..." << endl;
	vocab = bowtrainer->cluster();
	//vocab.convertTo(vocab,CV_8UC1);
	cout << "Done." << endl;

	// Save the vocabulary
	FileStorage fs("vocab.xml", FileStorage::WRITE);
	fs << "vocabulary" << vocab;
	fs.release();

	cout << "Built vocabulary" << endl;
}

void Classifier::load_vocab() {
	//load the vocabulary
	FileStorage fs("vocab.xml", FileStorage::READ);
	fs["vocabulary"] >> vocab;
	fs.release();

	// Set the vocabulary for the BOW descriptor extractor
	bowDescriptorExtractor->setVocabulary(vocab);
}
void Classifier::load_classifier() {
	rtree = new CvRTrees;
	rtree->load("rf.xml");
}

void ConvertCloudtoGrayMat(const PointCloudBgr &in, Mat &out) {
	out = Mat(in.height,in.width,CV_8UC1);
	unsigned char* pO = out.data;
	PointCloudBgr::const_iterator pI = in.begin();
	while(pI != in.end()) {
		*pO = Round(0.2126*pI->r + 0.7152*pI->g + 0.0722*pI->g);
		++pI; ++pO;
	}
}

void EstimateNormals(const PointCloud<PointXYZRGBA>::ConstPtr &cloud, PointCloud<PointNormal>::Ptr &normals, bool fill) {
	pcl::IntegralImageNormalEstimation<pcl::PointXYZRGBA, pcl::PointNormal> ne;
	ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
	ne.setMaxDepthChangeFactor(0.02f);
	ne.setNormalSmoothingSize(10.0f);
	ne.setInputCloud(cloud);
	ne.compute(*normals);
	if(fill) {
		PointCloudNormal::iterator p = normals->begin();
		while(p != normals->end()) {
			if(_isnan(p->normal_x))
				p->normal_x = 0;
			if(_isnan(p->normal_y))
				p->normal_y = 0;
			if(_isnan(p->normal_z))
				p->normal_z = 0;
			++p;
		}
	}
}

void trackbarChange(int pos, void *data) {
	AnnotationData *annotation = (AnnotationData*) data;
	annotation->Lock();
	annotation->currLabel = pos;
	annotation->Unlock();
}

void onMouse(int evt, int x, int y, int flags, void *data) {
	if(evt != EVENT_LBUTTONDOWN)
		return;

	AnnotationData *annotation = (AnnotationData*) data;
	annotation->Lock();
	//we need to get the segment id for the clicked segment
	int id = annotation->labelCloud(x,y).intensity;
	//we need to get the color for the current label
	Vec3b color = annotation->colors[annotation->currLabel];
	//Now we need to alpha multiply all the original image values with the new color in the new image for id
	Mat_<Vec3b>::iterator pOrig = annotation->orig.begin<Vec3b>();
	Mat_<Vec3b>::iterator pImg = annotation->img.begin<Vec3b>();
	Mat_<int>::iterator pLabel = annotation->label.begin<int>();
	PointCloudInt::const_iterator pCloud = annotation->labelCloud.begin();
	float added, newVal;
	while(pCloud != annotation->labelCloud.end()) {
		if(pCloud->intensity == id) {
			if(annotation->currLabel == 0)
				*pImg = *pOrig;
			else {
				for(int i = 0; i < 3; i++) {
					added = float(color[i]) * ALPHA;
					newVal = float((*pOrig)[i]) * (1.0f - ALPHA) + added;
					(*pImg)[i] = Round(newVal);
				}
			}
			*pLabel = annotation->currLabel;
		}
		++pCloud; ++pOrig; ++pImg; ++pLabel;
	}
	annotation->Unlock();
	imshow("Annotation",annotation->img);
	//imshow("Label",annotation->label);
}

void Classifier::Annotate() {

	for(boost::filesystem::directory_iterator i(direc), end_iter; i != end_iter; i++) {
		string filename = string(direc) + i->path().filename().string();
		AnnotationData data;
		pcl::io::loadPCDFile(filename,data.cloud);

		//SHGraphSegment(data.cloud,parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],&data.labelCloud,&data.segmentCloud);
		EstimateNormals(data.cloud.makeShared(),data.normals,false);
		SegmentColorAndNormals(data.cloud,data.normals,normalParameters[0],normalParameters[1],normalParameters[2],normalParameters[3],&data.labelCloud,&data.segmentCloud);
		GetMatFromCloud(data.cloud,data.orig);
		data.img = data.orig.clone();
		data.label = Mat::zeros(data.img.rows,data.img.cols,CV_32S);
		//red = floor, blue = structure, green = objects
		cout << "1 = RED = floor" << endl;
		cout << "2 = BLUE = structure" << endl;
		cout << "3 = GREEN = objects" << endl;
		imshow("Annotation",data.img);
		Mat segmentImg;
		GetMatFromCloud(data.segmentCloud,segmentImg);
		imshow("Segmentation",segmentImg);
		waitKey(1);
		int currLabel = 0;
		createTrackbar("Labels","Annotation",nullptr,data.NUM_LABELS,&trackbarChange,&data);
		setMouseCallback("Annotation",onMouse,&data);
		bool quit = false;
		while(!quit) {
			int c = waitKey(30);
			if((c & 255) == 27) {
				cout << "Done Annotating image: " << i->path().filename().string() << endl;
				quit = true;
			}
		}
		imwrite_depth(string(i->path().filename().string() + ".dep").c_str(), data.label);
	}
}

void Classifier::InitializeTesting() {
	load_vocab();
	load_classifier();
}

void Classifier::CreateAugmentedCloud(PointCloudBgr::Ptr &out) {
	*out = data.cloud;
	PointCloudBgr::iterator pO = out->begin();
	PointCloudInt::const_iterator pI = data.labelCloud.begin();
	float added, newVal;
	while(pO != out->end()) {
		Vec3b color = data.colors[Clamp(int(pI->intensity),0,3)];
		added = float(color[0]) * ALPHA;
		newVal = float(pO->b) * (1.0f - ALPHA) + added;
		pO->b = Round(newVal);
		added = float(color[1]) * ALPHA;
		newVal = float(pO->g) * (1.0f - ALPHA) + added;
		pO->g = Round(newVal);
		added = float(color[2]) * ALPHA;
		newVal = float(pO->r) * (1.0f - ALPHA) + added;
		pO->r = Round(newVal);
		++pO; ++pI;
	}
}