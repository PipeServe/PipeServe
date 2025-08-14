# PipeServe

PipeServe is a cost-efficient serverless LLM inference framework that employs a decoupled combinatorial optimization of adaptive batching and model partitioning.

## Overview

PipeServe integrates three main modules: Model Profiler, Batch Configurator, and Model Partitioner. Users first submit inference jobs to the Model Profiler, which includes a large model, SLOs. The Profiler analyzes the inference time and GPU memory consumption of different layers on GPUs. Leveraging the profiled job statistics above, the Batch Configurator then employs a backward bucket-scan to generate a cost-efficient batch configuration. It further utilizes a SLO-aware non-uniform model partitioning strategy in Model Partitioner to partition model layers into stages, aiming to minimize the pipeline stalls. Finally, the inference job is deployed with the cost-efficient batch configuration and model partitioning plan in the public serverless clouds.
![Architecture](images/architecture.png)



## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

- Python 3.11
- Torch 2.5.1
- Transformers 4.48.0
- llm-analysis>=0.2.2



### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/PipeServe/PipeServe.git
cd PipeServe
pip install -r requirements.txt
```