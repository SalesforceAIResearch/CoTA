# 🌮 TACO: Learning Multi-modal Action Models with Synthetic Chains-of-Thought-and-Action

<h3 align="left"> <a href="https://taco-project.github.io/">🌐 Website</a> | <a href="https://arxiv.org/pdf/2412.05479">📑 Arxiv</a> | <a href="https://huggingface.co/collections/agentstudio-family/cota-datasets-6765ba941a7a2d2b78f22553">🤗 Datasets</a></h3>
    
<h5 align="left"> If you like our project or are interested in its updates, please star us on GitHub :) Thank you! ⭐ </h2>

## News
 
 🔥 [2024-12-10]: Data released!

## How do we generate CoTA data?
We generate Chains-of-Thought-and–Action (CoTA) data automatically with two approaches as shown below: model-based generation (top) and programmatic generation (bottom).

<p align="center">
  <img src="data_gen.png" width="1000" style="margin-bottom: 0.2;"/>
  <p align="center">Figure 1. CoTA data generation method</p>
<p>

In model-based generation, we take existing image and QA pairs as inputs and prompt a large language model (i.e. GPT-4o) to generate
either a chain-of-thought-and-action (CoTA) or chain-of-thought (CoT) without actions to answer the questions. Then, we verify that the
chains lead to correct final answers and parse successfully; if not, we convert them into the direct answer (Direct) format with groundtruth
answers. In programmatic generation, we first annotate images with human labelers and models, and then use the dense annotations to fill
in manually written templates and generate QA and the corresponding CoTA with Python programs.

## Code usage
### Installation
You can easily download the repo and set up the environment via:
```
git clone https://github.com/airesearch-emu/cota.git
cd cota

pip install -r requirements.txt
```

### Model-based CoTA generation 
- Step 1: Modify the environment and code paths in the script ```scripts/generate_mm_trajs.sh```
- Step 2: Run the script ```generate_mm_trajs.sh $subset```, where ```$subset``` is a string representing a subset of the Cauldron dataset, or ```mantis-$subset``` for Mantis-Instruct.
For example, for Cauldron: ```generate_mm_trajs.sh ai2d```; for Mantis: ```generate_mm_trajs.sh mantis-contrastive_caption```.

### Programmatic CoTA generation 
- Generate CoTA for single-image examples:
```python cota/gen_tool_single.py```

- For multi-image examples:
```python cota/gen_tool_multi.py```


### License
The CoTA datasets are licensed under the noncommerical license [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Users need to make their own assessment regarding any obligations or responsibilities under the corresponding licenses or terms and conditions pertaining to the original datasets and data. This release is for research purposes only in support of an academic paper.

### Citation
Please cite us if you find our repository helpful. Thank you!
```
@misc{ma2024tacolearningmultimodalaction,
      title={TACO: Learning Multi-modal Action Models with Synthetic Chains-of-Thought-and-Action}, 
      author={Zixian Ma and Jianguo Zhang and Zhiwei Liu and Jieyu Zhang and Juntao Tan and Manli Shu and Juan Carlos Niebles and Shelby Heinecke and Huan Wang and Caiming Xiong and Ranjay Krishna and Silvio Savarese},
      year={2024},
      eprint={2412.05479},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.05479}, 
}
```
