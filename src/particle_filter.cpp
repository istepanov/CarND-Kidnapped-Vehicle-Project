/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>

#include "particle_filter.h"

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// DONE: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 1000;
	weights.resize(num_particles, 1.0);

	// This line creates a
	normal_distribution<double> dist_x(x, std[0]);				// normal (Gaussian) distribution for x
	normal_distribution<double> dist_y(y, std[1]);				// normal (Gaussian) distribution for y
	normal_distribution<double> dist_theta(theta, std[2]);		// normal (Gaussian) distribution for theta

	for (int i = 0; i < num_particles; ++i) {
		double sample_x = dist_x(gen);
		double sample_y = dist_y(gen);
		double sample_theta = dist_theta(gen);

		Particle particle;
		particle.id = i;
		particle.x = sample_x;
		particle.y = sample_y;
		particle.theta = sample_theta;
		particle.weight = 1.0;

		particles.push_back(particle);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// DONE: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	for (auto& p : particles) {
		if (fabs(yaw_rate) > 0.00001) {
			double velocity_by_yaw_rate = velocity/yaw_rate;
			double next_theta = p.theta + yaw_rate * delta_t;
			p.x += velocity_by_yaw_rate * (sin(next_theta) - sin(p.theta));
			p.y += velocity_by_yaw_rate * (cos(p.theta)  - cos(next_theta));
			p.theta = next_theta;
		} else {
			double distance = velocity * delta_t;
			p.x += distance * cos(p.theta);
			p.y += distance * sin(p.theta);
		}

		// add noise
		normal_distribution<double> dist_x(p.x, std_pos[0]);
		normal_distribution<double> dist_y(p.y, std_pos[1]);
		normal_distribution<double> dist_theta(p.theta, std_pos[2]);

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
	}
}

vector<LandmarkObs> ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {
	// DONE: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	vector<LandmarkObs> associated_landmarks;
	LandmarkObs closest;

	for (auto obs: observations){
		double shortest = numeric_limits<double>::max();
		for (auto pred: predicted){
			double distance = dist(obs.x, obs.y, pred.x, pred.y);
			if (distance < shortest) {
				shortest = distance;
				closest = pred;
			}
		}

		associated_landmarks.push_back(closest);
	}

	return associated_landmarks;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// DONE: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double sigma_x = std_landmark[0];
	double sigma_y = std_landmark[1];

	for (int i=0; i < particles.size(); ++i) {
	    Particle p = particles[i];

	    // transform observations from the particle coordinate system to the map system
		vector<LandmarkObs> transformed_observations;
		for (auto observation: observations){
			LandmarkObs transformed_observation;
			transformed_observation.x = p.x + observation.x * cos(p.theta) - observation.y * sin(p.theta);
			transformed_observation.y = p.y + observation.x * sin(p.theta) + observation.y * cos(p.theta);
			transformed_observation.id = observation.id;
			transformed_observations.push_back(transformed_observation);
		}

		// get all landmarks that are within sight of the particle
		vector<LandmarkObs> predicted;
		for (auto landmark: map_landmarks.landmark_list){
			double distance = dist(p.x, p.y, landmark.x_f, landmark.y_f);
			if (distance < sensor_range) {
				LandmarkObs one_landmark;
				one_landmark.id = landmark.id_i;
				one_landmark.x = landmark.x_f;
				one_landmark.y = landmark.y_f;
				predicted.push_back(one_landmark);
			}
		}

		// then associate the nearest landmark to every observation of the particle
		vector<LandmarkObs> associated_landmarks = dataAssociation(predicted, transformed_observations);
		double probability = 1;
		for (int j=0; j < associated_landmarks.size(); ++j){
			double dx = transformed_observations[j].x - associated_landmarks[j].x;
			double dy = transformed_observations[j].y - associated_landmarks[j].y;
			probability *= 1.0 / (2 * M_PI * sigma_x * sigma_y) * exp(-dx*dx / (2 * sigma_x * sigma_x)) * exp(-dy*dy / (2 * sigma_y * sigma_y));
		}

		p.weight = probability;
		weights[i] = probability;
	}
}

void ParticleFilter::resample() {
	// DONE: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	discrete_distribution<int> discrete(weights.begin(), weights.end());
	vector<Particle> weighted_sample(num_particles);

	for (int i = 0; i < num_particles; ++i) {
		int j = discrete(gen);
		weighted_sample[i] = particles[j];
	}

	particles = weighted_sample;
}

Particle ParticleFilter::SetAssociations(Particle particle, vector<int> associations, vector<double> sense_x, vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
