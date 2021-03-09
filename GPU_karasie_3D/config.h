#pragma once
#include <sstream>
#include <iostream>
#include <fstream>
#include <map>
#define MIN(a,b) ((a) < (b) ? (a) : (b))

using namespace std;

struct Config
{
	int fishesCount;				//number of fish in the simulation
	int maxFPS;						//limiting the number of frames, negative - unlimited
	int cubeSideLength;				//window length    
	int windowWidth;				//window width
	int windowHeight;				//window height
	float maxSpeed;					//maximum fish speed (distance in one frame)
	float maxAcceleration;			//maximum acceleration of the fish
	float radious;					//neighbors visibility distance
	float sociableProportion;       //
	float asocialProportion;        //proportion of fishTypes (1,1,2 = 25%,25%,50%)
	float crowdFollowingProportion; //
};

/// <summary>
/// Load config variables from "config.txt" file
/// </summary>
/// <returns>Config structure</returns>
Config configure()
{
	std::ifstream file("config.txt");

	if (!file)
	{
		file.close();
		cout << "Bad config file" << endl;
		exit(-1);
	}

	std::stringstream buffer;

	buffer << file.rdbuf();

	file.close();

	std::map<string, string> my_map;

	std::string line;
	while (std::getline(buffer, line))
	{
		std::istringstream is_line(line);
		std::string key;
		if (std::getline(is_line, key, '='))
		{
			std::string value;
			if (std::getline(is_line, value, ';'))
			{
				my_map[key] = value;
			}

		}
	}

	Config config{};
	config.fishesCount = stoi(my_map["fishesCount"]);
	config.maxFPS = stoi(my_map["maxFPS"]);
	config.windowWidth = stoi(my_map["width"]);
	config.cubeSideLength = stoi(my_map["cubeSideLength"]);
	config.windowHeight = stoi(my_map["height"]);
	config.maxSpeed = stof(my_map["maxSpeed"]);
	config.maxAcceleration = stof(my_map["maxAcceleration"]);
	config.sociableProportion = stof(my_map["sociableProportion"]);
	config.asocialProportion = stof(my_map["asocialProportion"]);
	config.crowdFollowingProportion = stof(my_map["crowdFollowingProportion"]);
	config.radious =  MIN(stof(my_map["radious"]),MIN(MIN(config.windowWidth, config.windowHeight), config.cubeSideLength)/2);
	


	return config;
}



cudaError_t setupCuda()
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

Error:

	return cudaStatus;
}