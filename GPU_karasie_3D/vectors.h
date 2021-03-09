#pragma once

struct color
{
	unsigned char R;
	unsigned char G;
	unsigned char B;
	unsigned char A;
};


struct float5
{
	float x;
	float y;
	float z;
	float d4;
	color color;
};


struct vectorI
{
	int x;
	int y;
};


struct triangle
{
	float5 a;
	float5 b;
	float5 c;
};

