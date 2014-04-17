#include "Labeling.h"

using namespace std;
using namespace pcl;
using namespace cv;

inline int GetClass(const PointCloudInt &cloud, const Mat &labels, int id) {
	int i, ret = 0;
	int *lookup = new int[NUM_LABELS];
	for(i = 0; i < NUM_LABELS; i++)
		lookup[i] = 0;
	PointCloudInt::const_iterator p = cloud.begin();
	Mat_<int>::const_iterator pL = labels.begin<int>();
	while(p != cloud.end()) {
		if(p->intensity == id)
			lookup[*pL]++;
		++p; ++pL;
	}
	int max = lookup[0], maxLoc = 0;
	for(i = 0; i < NUM_LABELS; i++) {
		if(lookup[i] > max) {
			max = lookup[i];
			maxLoc = i;
		}
	}
	try {
		delete[] lookup;
	} catch(...) {
		cout << "small error here" << endl;
	}
	return maxLoc;
}

inline float HistTotal(LABXYZUVW *hist) {
	float tot = 0.0f;
	for(int k = 0; k < NUM_BINS; k++) {
		tot += hist[k].u;
	}
	return tot;
}

inline void CalcMask(const PointCloudInt &cloud, int id, Mat &mask) {
	PointCloudInt::const_iterator pC = cloud.begin();
	uchar *pM = mask.data;
	while(pC != cloud.end()) {
		if(pC->intensity == id)
			*pM = 255;
		++pM; ++pC;
	}
}

void GetFeatureVectors(Mat &trainData, Classifier &cl, const RegionTree3D &tree, const PointCloudBgr &cloud, const PointCloudInt &labelCloud, const Mat &label, const int numImage) {
	//for each top level region, I need to give it a class name.
	int k;
	const int size1 = 14 + 6*NUM_BINS + 3*NUM_BINS_XYZ, size2 = (size1 + NUM_CLUSTERS + 3);
	Mat gImg, desc;
	ConvertCloudtoGrayMat(cloud,gImg);
	vector<KeyPoint> kp;
	cl.featureDetector->detect(gImg,kp);
	vector<KeyPoint> regionPoints;
	regionPoints.reserve(kp.size());
	vector<Region3D*>::const_iterator p = tree.top_regions.begin();
	//Mat element = getStructuringElement(MORPH_RECT, Size( 2*2 + 1, 2*2+1 ), Point( 2, 2 ) );
	for(int i = 0; i < tree.top_regions.size(); i++, p++) {
		////Calculate mask
		//Mat desc, mask = Mat::zeros(img.size(),CV_8UC1);
		//CalcMask(cloud,(*p)->m_centroid3D.intensity,mask);
		//dilate(mask,mask,element);
		////get features
		//cl.CalculateBOWFeatures(img,mask,desc);
		Mat desc;
		regionPoints.clear();
		vector<KeyPoint>::iterator pK = kp.begin();
		while(pK != kp.end()) {
			PointXYZI p3D = labelCloud(pK->pt.x,pK->pt.y);
			if(p3D.x >= (*p)->m_min3D.x && p3D.x <= (*p)->m_max3D.x && p3D.y >= (*p)->m_min3D.y && p3D.y <= (*p)->m_max3D.y && p3D.z >= (*p)->m_min3D.z && p3D.z <= (*p)->m_max3D.z)
				regionPoints.push_back(*pK);
			++pK;
		}
		cl.bowDescriptorExtractor->compute(gImg,regionPoints,desc);
		if(desc.empty())
			desc = Mat::zeros(1,NUM_CLUSTERS,CV_32F);
		int id = GetClass(labelCloud,label,(*p)->m_centroid3D.intensity);
		if(id != 0) {
			Mat vec = Mat(1,size2,CV_32F);
			float *pV = (float*)vec.data;
			*pV++ = float((*p)->m_size);
			*pV++ = (*p)->m_centroid.x;
			*pV++ = (*p)->m_centroid.y;
			if(!boost::math::isnan<float>((*p)->m_centroid3D.z) && !boost::math::isinf<float>((*p)->m_centroid3D.z)) {
				*pV++ = (*p)->m_centroid3D.x;
				*pV++ = (*p)->m_centroid3D.y;
				*pV++ = (*p)->m_centroid3D.z;
			} else {
				*pV++ = 0;
				*pV++ = 0;
				*pV++ = 0;
			}
			float a = ((*p)->m_max3D.x - (*p)->m_min3D.x), b = ((*p)->m_max3D.y - (*p)->m_min3D.y), c = ((*p)->m_max3D.z - (*p)->m_min3D.z);
			if(!boost::math::isnan<float>((*p)->m_min3D.z) && !boost::math::isinf<float>((*p)->m_min3D.z)) {
				*pV++ = (*p)->m_min3D.x;
				*pV++ = (*p)->m_min3D.y;
				*pV++ = (*p)->m_min3D.z;
			} else {
				*pV++ = 0;
				*pV++ = 0;
				*pV++ = 0;
			}
			if(!boost::math::isnan<float>((*p)->m_max3D.z) && !boost::math::isinf<float>((*p)->m_max3D.z)) {
				*pV++ = (*p)->m_max3D.x;
				*pV++ = (*p)->m_max3D.y;
				*pV++ = (*p)->m_max3D.z;
			} else {
				*pV++ = 0;
				*pV++ = 0;
				*pV++ = 0;
			}
			if(!boost::math::isnan<float>(b) && !boost::math::isinf<float>(b)) {
				*pV++ = sqrt(a*a + c*c);
				*pV++ = b;
			} else {
				*pV++ = 0;
				*pV++ = 0;
			}
			//LABXYZUVW *p1 = (*p)->m_hist;
			//float tot = HistTotal((*p)->m_hist);
			for(k = 0; k < NUM_BINS; k++)
				*pV++ = float((*p)->m_hist[k].a)/(*p)->m_size;
			for(k = 0; k < NUM_BINS; k++)
				*pV++ = float((*p)->m_hist[k].b)/(*p)->m_size;
			for(k = 0; k < NUM_BINS; k++)
				*pV++ = float((*p)->m_hist[k].l)/(*p)->m_size;
			for(k = 0; k < NUM_BINS; k++)
				*pV++ = (*p)->m_hist[k].u/(*p)->m_size;
			for(k = 0; k < NUM_BINS; k++)
				*pV++ = (*p)->m_hist[k].v/(*p)->m_size;
			for(k = 0; k < NUM_BINS; k++)
				*pV++ = (*p)->m_hist[k].w/(*p)->m_size;
			for(k = 0; k < NUM_BINS_XYZ; k++)
				*pV++ = (*p)->m_hist[k].x/(*p)->m_size;
			for(k = 0; k < NUM_BINS_XYZ; k++)
				*pV++ = (*p)->m_hist[k].y/(*p)->m_size;
			for(k = 0; k < NUM_BINS_XYZ; k++)
				*pV++ = (*p)->m_hist[k].z/(*p)->m_size;
			float *pD = (float*)desc.data;
			for(k = 0; k < desc.cols; k++, pD++)
				*pV++ = *pD;
			*pV++ = float((*p)->m_centroid3D.intensity);
			*pV++ = float(numImage);
			*pV++ = float(id);
			trainData.push_back(vec);
		}
	}
}

void GetMatFromRegion(Region3D *reg, Classifier &cl, const Mat &gImg, const PointCloudInt &labelCloud, vector<KeyPoint> &kp, vector<float> &sample, int sample_size) {
	int k;
	sample.resize(sample_size);
	//Calculate mask
	//Mat desc, mask = Mat::zeros(img.size(),CV_8UC1);
	//CalcMask(cloud,reg->m_centroid3D.intensity,mask);
	////get features
	//cl.CalculateBOWFeatures(img,mask,desc);
	Mat desc;
	vector<KeyPoint> regionPoints;
	regionPoints.reserve(kp.size());
	vector<KeyPoint>::iterator pK = kp.begin();
	while(pK != kp.end()) {
		PointXYZI p3D = labelCloud(pK->pt.x,pK->pt.y);
		if(p3D.x >= reg->m_min3D.x && p3D.x <= reg->m_max3D.x && p3D.y >= reg->m_min3D.y && p3D.y <= reg->m_max3D.y && p3D.z >= reg->m_min3D.z && p3D.z <= reg->m_max3D.z)
			regionPoints.push_back(*pK);
		++pK;
	}
	cl.bowDescriptorExtractor->compute(gImg,regionPoints,desc);
	if(desc.empty())
		desc = Mat::zeros(1,NUM_CLUSTERS,CV_32F);
	vector<float>::iterator p = sample.begin();
	*p++ = float(reg->m_size);
	*p++ = reg->m_centroid.x;
	*p++ = reg->m_centroid.y;
	if(!boost::math::isnan<float>(reg->m_centroid3D.z) && !boost::math::isinf<float>(reg->m_centroid3D.z)) {
		*p++ = reg->m_centroid3D.x;
		*p++ = reg->m_centroid3D.y;
		*p++ = reg->m_centroid3D.z;
	} else {
		*p++ = 0;
		*p++ = 0;
		*p++ = 0;
	}
	float a = (reg->m_max3D.x - reg->m_min3D.x), b = (reg->m_max3D.y - reg->m_min3D.y), c = (reg->m_max3D.z - reg->m_min3D.z);
	if(!boost::math::isnan<float>(reg->m_min3D.z) && !boost::math::isinf<float>(reg->m_min3D.z)) {
		*p++ = reg->m_min3D.x;
		*p++ = reg->m_min3D.y;
		*p++ = reg->m_min3D.z;
	} else {
		*p++ = 0;
		*p++ = 0;
		*p++ = 0;
	}
	if(!boost::math::isnan<float>(reg->m_max3D.z) && !boost::math::isinf<float>(reg->m_max3D.z)) {
		*p++ = reg->m_max3D.x;
		*p++ = reg->m_max3D.y;
		*p++ = reg->m_max3D.z;
	} else {
		*p++ = 0;
		*p++ = 0;
		*p++ = 0;
	}
	if(!boost::math::isnan<float>(b) && !boost::math::isinf<float>(b)) {
		*p++ = sqrt(a*a+c*c);
		*p++ = b;
	} else {
		*p++ = 0;
		*p++ = 0;
	}
	for(k = 0; k < NUM_BINS; k++)
		*p++ = reg->m_hist[k].a / reg->m_size;
	for(k = 0; k < NUM_BINS; k++)
		*p++ = reg->m_hist[k].b / reg->m_size;
	for(k = 0; k < NUM_BINS; k++)
		*p++ = reg->m_hist[k].l / reg->m_size;
	for(k = 0; k < NUM_BINS; k++)
		*p++ = reg->m_hist[k].u / reg->m_size;
	for(k = 0; k < NUM_BINS; k++)
		*p++ = reg->m_hist[k].v / reg->m_size;
	for(k = 0; k < NUM_BINS; k++)
		*p++ = reg->m_hist[k].w / reg->m_size;
	for(k = 0; k < NUM_BINS_XYZ; k++)
		*p++ = reg->m_hist[k].x / reg->m_size;
	for(k = 0; k < NUM_BINS_XYZ; k++)
		*p++ = reg->m_hist[k].y / reg->m_size;
	for(k = 0; k < NUM_BINS_XYZ; k++)
		*p++ = reg->m_hist[k].z / reg->m_size;
	float *pD = (float*)desc.data;
	for(k = 0; k < desc.cols; k++, pD++)
		*p++ = *pD;
}

void BuildDataset(string direc) {
	srand(time(NULL));
	PointCloudBgr cloud,segment;
	PointCloudInt labelCloud;
	Mat label, trainData;
	boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > normals(new pcl::PointCloud<pcl::PointNormal>);
	Classifier c(direc);
	c.load_vocab();
	//open training file
	/*FILE *fp = fopen("features.txt","wb");
	if(fp == NULL)
	throw exception("Couldn't open features file");
	fprintf(fp,"size,cx,cy,c3x,c3y,c3z,minx,miny,minz,maxx,maxy,maxz,xdist,ydist");
	for(int j = 0; j < 9; j++) {
	for(int k = 0; k < (j < 6 ? NUM_BINS : NUM_BINS_XYZ); k++) {
	fprintf(fp,",h%d_%d",j,k);
	}
	}
	fprintf(fp,",frame,class\n");*/
	int count = 0;
	string folder;
	for(boost::filesystem::directory_iterator i(direc), end_iter; i != end_iter; i++) {
		string filename = string(direc) + i->path().filename().string();
		pcl::io::loadPCDFile(filename,cloud);
		label = imread_depth(string("labels/" + i->path().filename().string() + ".dep").c_str());
		EstimateNormals(cloud.makeShared(),normals,false);
		int segments = SegmentColorAndNormals(cloud,normals,normalParameters[0],normalParameters[1],normalParameters[2],normalParameters[3],&labelCloud,&segment);
		RegionTree3D tree;
		tree.Create(cloud,labelCloud,*normals,segments,0);
		tree.PropagateRegionHierarchy(normalParameters[4]);
		tree.ImplementSegmentation(normalParameters[5]);

		GetFeatureVectors(trainData,c,tree,cloud,labelCloud,label,count);
		stringstream num;
		num << "training/" << count << ".flt";
		imwrite_float(num.str().c_str(),trainData);
		count++;

		//release stuff
		segment.clear();
		cloud.clear();
		labelCloud.clear();
		label.release();
		trainData.release();
		normals->clear();
		tree.top_regions.clear();
		tree.Release();
	}
	FileStorage tot("count.yml", FileStorage::WRITE);
	tot << "count" << count;
	//fclose(fp);
	tot.release();
}

void BuildRFClassifier(string direc) {
	Classifier c(direc);
	FileStorage fs("count.yml", FileStorage::READ);
	int i,count;
	fs["count"] >> count;
	fs.release();
	Mat data, train, labels;
	for(i = 0; i < count; i++) {
		Mat tmp;
		stringstream num;
		num << "training/" << i << ".flt";
		tmp = imread_float(num.str().c_str());
		data.push_back(tmp);
	}
	train = data.colRange(0,data.cols-3);
	labels = data.col(data.cols-1);
	labels.convertTo(labels,CV_32S);

	// define all the attributes as numerical
	// alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
	// that can be assigned on a per attribute basis
	Mat var_type = Mat(train.cols + 1, 1, CV_8U );
	var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical
	// this is a classification problem (i.e. predict a discrete number of class
	// outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL
	var_type.at<uchar>(train.cols, 0) = CV_VAR_CATEGORICAL;
	//float priors[] = {1,1};
	CvRTParams params = CvRTParams(25, // max depth
		5, // min sample count
		0, // regression accuracy: N/A here
		false, // compute surrogate split, no missing data
		15, // max number of categories (use sub-optimal algorithm for larger numbers)
		nullptr, // the array of priors
		false,  // calculate variable importance
		16,       // number of variables randomly selected at node and used to find the best split(s).
		100,	 // max number of trees in the forest
		0.01f,				// forrest accuracy
		CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
		);

	// train random forest classifier (using training data)
	CvRTrees* rtree = new CvRTrees;

	rtree->train(train, CV_ROW_SAMPLE, labels,
		Mat(), Mat(), var_type, Mat(), params);
	rtree->save("rf.xml");
	delete rtree;
}

void Classifier::TestCloud(const PointCloudBgr &cloud) {
	data.cloud = cloud;
	EstimateNormals(cloud.makeShared(),data.normals,false);
	int segments = SegmentColorAndNormals(cloud,data.normals,normalParameters[0],normalParameters[1],normalParameters[2],normalParameters[3],&data.labelCloud,&data.segmentCloud);
	RegionTree3D tree;
	tree.Create(cloud,data.labelCloud,*data.normals,segments,0);
	//tree.PropagateRegionHierarchy(normalParameters[4]);
	//tree.ImplementSegmentation(normalParameters[5]);

	int result, feature_len = 14 + 6*NUM_BINS + 3*NUM_BINS_XYZ + NUM_CLUSTERS;
	vector<KeyPoint> kp;
	Mat gImg;
	ConvertCloudtoGrayMat(cloud,gImg);
	featureDetector->detect(gImg,kp);
	//Mat element = getStructuringElement(MORPH_RECT, Size( 2*2 + 1, 2*2+1 ), Point( 2, 2 ) );
	//vector<Region3D*>::const_iterator p = tree.top_regions.begin();
	vector<Region3D*>::const_iterator p = tree.region_list.begin();
	//for(int i = 0; i < tree.top_regions.size(); i++, p++) {
	for(int i = 0; i < tree.region_list.size(); i++, p++) {
		if(!boost::math::isnan<float>((*p)->m_centroid3D.z) && !boost::math::isinf<float>((*p)->m_centroid3D.z) && abs((*p)->m_centroid3D.z) > 0.01) {
			vector<float> sample;
			GetMatFromRegion(*p,*this,gImg,data.labelCloud,kp,sample,feature_len);
			Mat sampleMat = Mat(sample);
			result = Round(rtree->predict(sampleMat));
			tree.SetBranch(*p,0,result);
		} else {
			tree.SetBranch(*p,0,0);
		}
	}

	//release stuff
	data.normals->clear();
	tree.top_regions.clear();
	tree.Release();
}
