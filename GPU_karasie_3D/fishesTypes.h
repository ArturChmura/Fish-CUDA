#pragma once
// types of fihses avalible 
const unsigned char sociable = 0;
const unsigned char asocial = 1;
const unsigned char crowdFollowing = 2;

// colors of fihses types 
__device__ const color sociableColor = { (unsigned char)248,  (unsigned char)252, (unsigned char)15, (unsigned char)255 };
__device__ const color asocialColor = { (unsigned char)0,  (unsigned char)0,  (unsigned char)0,  (unsigned char)255 };
__device__ const color crowdFollowingColor = { (unsigned char)230,  (unsigned char)0,  (unsigned char)255,  (unsigned char)255 };
__device__ const color frontColor = { (unsigned char)0, (unsigned char)255, (unsigned char)0, (unsigned char)255 };

const int typesCount = 3;