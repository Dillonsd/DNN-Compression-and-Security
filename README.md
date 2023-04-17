# DNN Compression and Security

This repository holds the code-base behind the "DNN Compression and Security" paper.

## Abstract

As machine learning (ML) models are increasingly embedded in everyday devices, the security of these models becomes increasingly important. In particular, the growing use of compression techniques in embedded ML raises questions about the potential impact of these techniques on backdoor attacks and backdoor detection. This paper addresses these issues by studying the impact of compression on poisoning-based backdoor attacks and backdoor detection, as well as the impact of backdooring on compression techniques. Additionally, we look at a novel attack vector that focuses on the interaction between compression and security. Our experiments demonstrate that compression can affect the effectiveness of backdoor attacks, depending on the specific compression technique used - and the possibility of using compression as a defense. We work towards this novel attack and propose future research directions that could solve this problem. These results underscore the need for continued research on the security of embedded ML and the impact of compression techniques on backdoor attacks and detection.

## Project Structure

Each part of the project is self-contained in its own folder. The folders are:

- `BackdoorDetection`: Contains the code for the backdoor detection experiments.
- `Badnets`: Contains the code for the backdoor poisoning experiments.
- `CompressionAwarePoisoning`: Contains the code for the compression-aware poisoning experiments.

## Usage

Each folder contains a `README.md` file with instructions on how to run the experiments.
The `requirements.txt` file contains the list of dependencies for each experiment.

To install the dependencies, run:

```bash
pip install -r requirements.txt
```
