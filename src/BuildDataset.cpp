#include "Labeling.h"

const float parameters[] = { 0.5f,500.0f,300,0.8f,500.0f,300,300,0.3f };

using namespace std;
using namespace pcl;
using namespace cv;

inline void EstimateNormals(const PointCloud<PointXYZRGBA>::ConstPtr &cloud, PointCloud<PointNormal>::Ptr &normals, bool fill) {
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
			*pV++ = (*p)->m_centroid3D.x;
			*pV++ = (*p)->m_centroid3D.y;
			*pV++ = (*p)->m_centroid3D.z;
			float a = ((*p)->m_max3D.x - (*p)->m_min3D.x), b = ((*p)->m_max3D.y - (*p)->m_min3D.y), c = ((*p)->m_max3D.z - (*p)->m_min3D.z);
			*pV++ = (*p)->m_min3D.x;
			*pV++ = (*p)->m_min3D.y;
			*pV++ = (*p)->m_min3D.z;
			*pV++ = (*p)->m_max3D.x;
			*pV++ = (*p)->m_max3D.y;
			*pV++ = (*p)->m_max3D.z;
			*pV++ = sqrt(a*a + c*c);
			*pV++ = b;
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

void GetMatFromRegion(Region3D *reg, Classifier &cl, const PointCloudBgr &cloud, const PointCloudInt &labelCloud, vector<KeyPoint> &kp, vector<float> &sample, int sample_size) {
	int k;
	sample.resize(sample_size);
	//Calculate mask
	//Mat desc, mask = Mat::zeros(img.size(),CV_8UC1);
	//CalcMask(cloud,reg->m_centroid3D.intensity,mask);
	////get features
	//cl.CalculateBOWFeatures(img,mask,desc);
	Mat gImg, desc;
	vector<KeyPoint> regionPoints;
	regionPoints.reserve(kp.size());
	vector<KeyPoint>::iterator pK = kp.begin();
	while(pK != kp.end()) {
		PointXYZI p3D = labelCloud(pK->pt.x,pK->pt.y);
		if(p3D.x >= reg->m_min3D.x && p3D.x <= reg->m_max3D.x && p3D.y >= reg->m_min3D.y && p3D.y <= reg->m_max3D.y && p3D.z >= reg->m_min3D.z && p3D.z <= reg->m_max3D.z)
			regionPoints.push_back(*pK);
		++pK;
	}
	ConvertCloudtoGrayMat(cloud,gImg);
	cl.descriptorExtractor->compute(gImg,regionPoints,desc);
	if(desc.empty())
		desc = Mat::zeros(1,NUM_CLUSTERS,CV_32F);
	vector<float>::iterator p = sample.begin();
	*p++ = float(reg->m_size);
	*p++ = reg->m_centroid.x;
	*p++ = reg->m_centroid.y;
	*p++ = reg->m_centroid3D.x;
	*p++ = reg->m_centroid3D.y;
	*p++ = reg->m_centroid3D.z;
	float a = (reg->m_max3D.x - reg->m_min3D.x), b = (reg->m_max3D.y - reg->m_min3D.y), c = (reg->m_max3D.z - reg->m_min3D.z);
	*p++ = reg->m_min3D.x;
	*p++ = reg->m_min3D.y;
	*p++ = reg->m_min3D.z;
	*p++ = reg->m_max3D.x;
	*p++ = reg->m_max3D.y;
	*p++ = reg->m_max3D.z;
	*p++ = sqrt(a*a+c*c);
	*p++ = b;
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

inline void GetMatFromCloud(const PointCloudBgr &cloud, Mat &img) {
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

inline void GetMatFromCloud(const PointCloudInt &cloud, Mat &img) {
	img = Mat(cloud.height,cloud.width,CV_32S);
	Mat_<int>::iterator pI = img.begin<int>();
	PointCloudInt::const_iterator pC = cloud.begin();
	while(pC != cloud.end()) {
		*pI = pC->intensity;
		++pI; ++pC;
	}
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
		//CreateLabeledCloudFromNYUPointCloud(cloud,label,&labelCloud);
		int segments = SHGraphSegment(cloud,parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],&labelCloud,&segment);
		EstimateNormals(cloud.makeShared(),normals,false);
		RegionTree3D tree;
		tree.Create(cloud,labelCloud,*normals,segments,0);
		tree.PropagateRegionHierarchy(parameters[6]);
		tree.ImplementSegmentation(parameters[7]);

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

//void TestRFClassifier(string direc) {
//	PointCloudBgr cloud,segment;
//	PointCloudInt labelCloud;
//	Mat img, depth, label;
//	boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > normals(new pcl::PointCloud<pcl::PointNormal>);
//	//open training file
//	Classifier c(direc);
//	c.load_vocab();
//	c.load_classifier();
//
//	int segments = SHGraphSegment(cloud,parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],&labelCloud,&segment);
//	EstimateNormals(cloud.makeShared(),normals,false);
//	RegionTree3D tree;
//	tree.Create(cloud,labelCloud,*normals,segments,0);
//	tree.PropagateRegionHierarchy(parameters[6]);
//	tree.ImplementSegmentation(parameters[7]);
//
//	int result, feature_len = 14 + 6*NUM_BINS + 3*NUM_BINS_XYZ + NUM_CLUSTERS;
//	Mat gImg;
//	cvtColor(img, gImg, CV_BGR2GRAY);
//	vector<KeyPoint> kp;
//	c.featureDetector->detect(gImg,kp);
//	//Mat element = getStructuringElement(MORPH_RECT, Size( 2*2 + 1, 2*2+1 ), Point( 2, 2 ) );
//	vector<Region3D*>::const_iterator p = tree.top_regions.begin();
//	for(int i = 0; i < tree.top_regions.size(); i++, p++) {
//		vector<float> sample;
//		GetMatFromRegion(*p,c,labelCloud,kp,img,sample,feature_len);
//		Mat sampleMat = Mat(sample);
//		result = Round(rtree->predict(sampleMat));
//		tree.SetBranch(*p,0,result);
//	}
//
//	Mat myResult, groundTruth, myResultColor, groundTruthColor, labelColor, segmentMat;
//	myResult = Mat(label.rows,label.cols,label.type());
//	groundTruth = Mat(label.rows,label.cols,label.type());
//	PointCloudInt::iterator pC = labelCloud.begin();
//	int *pNewL = (int*)groundTruth.data;
//	int *pNewC = (int*)myResult.data;
//	int *pL = (int *)label.data;
//	while(pC != labelCloud.end()) {
//		*pNewL = *pL;
//		*pNewC = pC->intensity;				
//		++pL; ++pC; ++pNewL; ++pNewC;
//	}
//	/*GetMatFromCloud(segment,segmentMat);
//	groundTruth.convertTo(groundTruth,CV_8UC1,63,0);
//	myResult.convertTo(myResult,CV_8UC1,63,0);
//	label.convertTo(labelColor,CV_8UC1,894,0);
//	applyColorMap(groundTruth,groundTruthColor,COLORMAP_JET);
//	applyColorMap(myResult,myResultColor,COLORMAP_JET);
//	imshow("color",img);
//	imshow("original label",labelColor);
//	imshow("label",groundTruthColor);
//	imshow("result",myResultColor);
//	imshow("segment",segmentMat);
//	waitKey();*/
//
//	//release stuff
//	segmentMat.release();
//	myResult.release();
//	groundTruth.release();
//	myResultColor.release();
//	groundTruthColor.release();
//	segment.clear();
//	cloud.clear();
//	labelCloud.clear();
//	img.release();
//	depth.release();
//	label.release();
//	normals->clear();
//	tree.top_regions.clear();
//	tree.Release();
//
//}