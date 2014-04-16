/*
Copyright (C) 2014 Steven Hickson

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA

*/

#include "Labeling.h"

using namespace std;
using namespace pcl;
using namespace cv;


inline void MakeCloudDense(PointCloud<PointXYZRGBA> &cloud) {
	PointCloud<PointXYZRGBA>::iterator p = cloud.begin();
	cloud.is_dense = true;
	for(int j = 0; j < cloud.height; j++) {
		for(int i = 0; i < cloud.width; i++) {
			if(_isnan(p->z)) {
				p->x = float(((float)i - KINECT_CX_D) * KINECT_FX_D);
				p->y = float(((float)j - KINECT_CY_D) * KINECT_FY_D);
				p->z = 0;
			}
			//p->a = 255;
			++p;
		}
	}
}

inline void MakeCloudDense(PointCloud<PointNormal>::Ptr &cloud) {
	PointCloud<PointNormal>::iterator p = cloud->begin();
	cloud->is_dense = true;
	for(int j = 0; j < cloud->height; j++) {
		for(int i = 0; i < cloud->width; i++) {
			if(_isnan(p->z)) {
				p->x = float(((float)i - KINECT_CX_D) * KINECT_FX_D);
				p->y = float(((float)j - KINECT_CY_D) * KINECT_FY_D);
				p->z = 0;
				p->normal_x = p->normal_y = p->normal_z = 0;

			}
			//p->a = 255;
			++p;
		}
	}
}

class SimpleSegmentViewer
{
public:
	SimpleSegmentViewer () : 
		label(new pcl::PointCloud<pcl::PointXYZI>), segment(new pcl::PointCloud<pcl::PointXYZRGBA>), sharedCloud(new pcl::PointCloud<pcl::PointXYZRGBA>), normals (new pcl::PointCloud<pcl::PointNormal>), update(false) { c = new Classifier(""); };
	void cloud_cb_ (const boost::shared_ptr<const PointCloud<PointXYZRGBA> > &cloud)
	{
		if(!cloud->empty()) {
			normalMutex.lock();
			double begin = pcl::getTime();
			copyPointCloud(*cloud,*sharedCloud);
			EstimateNormals(sharedCloud,normals);
			MakeCloudDense(*sharedCloud);
			//MakeCloudDense(normals);
			stseg.AddSlice(*sharedCloud,0.5f,900,500,0.8f,900,500,label,segment);
			//SegmentNormals(*sharedCloud,normals,0.5f,50,50,label,segment);
			label->clear();
			double end = pcl::getTime();
			update = true;
			normalMutex.unlock();
		}
	}

	void cloud_cb2_ (const boost::shared_ptr<const PointCloud<PointXYZRGBA> > &cloud)
	{
		if(!cloud->empty()) {
			normalMutex.lock();
			double begin = pcl::getTime();
			copyPointCloud(*cloud,*sharedCloud);
			c->TestCloud(*cloud);
			c->CreateAugmentedCloud(sharedCloud);
			double end = pcl::getTime();
			cout << "Time: " << (end - begin) << endl;
			update = true;
			Mat tmp;
			GetMatFromCloud(*sharedCloud,tmp);
			imshow("Results",tmp);
			waitKey(1);
			normalMutex.unlock();
		}
	}

	void run ()
	{
		// create a new grabber for OpenNI devices
		pcl::Grabber* my_interface = new pcl::Microsoft2Grabber();

		// make callback function from member function
		boost::function<void (const boost::shared_ptr<const PointCloud<PointXYZRGBA> >&)> f =
			boost::bind (&SimpleSegmentViewer::cloud_cb2_, this, _1);

		my_interface->registerCallback (f);

		//viewer.setBackgroundColor(0.0, 0.0, 0.5);
		c->InitializeTesting();
		my_interface->start ();

		bool finished = false;
		while(1);
		//while (!viewer.wasStopped())
		//{
		//	normalMutex.lock();
		//	if(update) {
		//		//viewer.removePointCloud("cloud");
		//		viewer.removePointCloud("original");
		//		viewer.addPointCloud(segment,"original");
		//		//viewer.addPointCloudNormals<pcl::PointXYZRGBA,pcl::PointNormal>(sharedCloud, normals);
		//		update = false;
		//		sharedCloud->clear();
		//		segment->clear();
		//		//normals->clear();
		//	}
		//	viewer.spinOnce();
		//	normalMutex.unlock();
		//}

		my_interface->stop ();
	}

	boost::shared_ptr<PointCloud<PointXYZI> > label;
	boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA> > segment, sharedCloud;
	boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > normals;
	//pcl::visualization::PCLVisualizer viewer;
	bool update;
	boost::mutex normalMutex;
	Segment3D stseg;
	Classifier *c;
};

int main (int argc, char** argv) {
	try {
		int run = atoi(argv[2]);
		if(run == 0) {
			Classifier c(argv[1]);
			c.build_vocab();
		} else if(run == 1) {
			Classifier c(argv[1]);
			c.Annotate();
		} else if(run == 2)
			BuildDataset(string(argv[1]));
		else if(run == 3)
			BuildRFClassifier(string(argv[1]));
		else {
			SimpleSegmentViewer v;
			v.run();
		}
	} catch (pcl::PCLException e) {
		cout << e.detailedMessage() << endl;
	} catch (std::exception &e) {
		cout << e.what() << endl;
	}
	cin.get();
	return 0;
}