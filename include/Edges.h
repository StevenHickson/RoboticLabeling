/*
Copyright (C) 2012 Steven Hickson

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

#ifndef EDGES_H
#define EDGES_H

#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"
#include <assert.h>

#define KINECT_CX_D 2.555704e+02
#define KINECT_CY_D 1.999361e+02
#define KINECT_FX_D 2.759283678919818e-03
#define KINECT_FY_D 2.759904330676281e-03

template <typename T> inline T Clamp(T a, T minn, T maxx)
{ return (a < minn) ? minn : ( (a > maxx) ? maxx : a ); }

inline int Round (float a)  
{
	assert( !_isnan( a ) );
	return static_cast<int>(a>=0 ? a+0.5f : a-0.5f); 
} 

typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudBgr;
typedef pcl::PointCloud<pcl::PointXYZI> PointCloudInt;
typedef pcl::PointCloud<pcl::PointNormal> PointCloudNormal;


class Edge {
public:
	float w;
	int a, b;
	bool valid;
	Edge() : w(0), a(0), b(0), valid(false) { };

	bool operator< (const Edge &other) const {
		return (w < other.w);
	}
};

class Edge3D : public Edge {
public:
	float w2;

	Edge3D() :  Edge(), w2(0) { };
};


#endif