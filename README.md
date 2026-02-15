Plant Disease Detector with AI Image Validation System

A secure deep learning–based system that detects plant diseases while verifying image authenticity using a two-stage AI pipeline. This ensures reliable predictions by preventing AI-generated or manipulated images from influencing disease classification.

Live Website

https://pyworkergj.github.io/PBL_2026_Garv-Jhuremalani_2427030363/

Overview

Traditional plant disease detection systems assume that all input images are authentic. However, AI-generated images can mislead models and produce incorrect predictions. This project introduces an authenticity verification stage before disease classification to ensure trustworthy and secure results.

The system uses ResNet-18 Convolutional Neural Networks to first detect whether an image is real or fake. Only authentic images proceed to the disease classification stage.

Features

Fake image detection using deep learning

Plant disease classification using ResNet-18 CNN

Two-stage secure AI pipeline

Real-time image input support

Treatment recommendation system

Interactive web-based presentation interface

Fully deployed using GitHub Pages

System Pipeline

Image Input
→ Fake Image Detector
→ Disease Classifier
→ Recommendation Engine
→ Output Result

Technologies Used

Frontend

HTML

Tailwind CSS

JavaScript

Chart.js

Machine Learning

Python

PyTorch

ResNet-18 CNN

Deployment

GitHub Pages

Dataset

PlantVillage Dataset

Synthetic fake images generated using Stable Diffusion

Results

Disease Classification Accuracy: 96.4%

Improved reliability by filtering fake images

Secure and scalable architecture

Literature Basis

Based on prior research including:

Mohanty et al. (2016) — CNN-based plant disease detection

Ferentinos (2018) — Deep learning architectures for plant classification

This project improves upon existing systems by integrating authenticity verification.

Repository Structure
docs/
│
├── index.html
├── mujlogo.jpg

Author

Garv Jhuremalani
B.Tech Computer Science and Engineering
Manipal University Jaipur

Registration Number: 2427030363

Guide

Dr. Ashok Kumar Saini
Department of Computer Science and Engineering
Manipal University Jaipur

Purpose

This project was developed as part of the Problem Based Learning (PBL) academic curriculum.
