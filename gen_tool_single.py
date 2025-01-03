import argparse
import os
from pprint import pprint

from pcota.base import *
from pcota.generators.tool.attribute import *
from pcota.generators.tool.object import *
from pcota.generators.tool.object_depth import *
from pcota.generators.tool.relation import *
from pcota.generators.tool.scene_graph import *

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--vg_dir', type=str, default="sample_data/")
	parser.add_argument('--n_works', type=int, default=12)
	parser.add_argument('--metadata_name', type=str, default="v2-tool")
	parser.add_argument('--plan_name', type=str, default="v2-tool")
	parser.add_argument('--name', type=str, default="v2-tool")
	parser.add_argument('--force', action="store_true", default=False)
	args = parser.parse_args()

	vg_dir = args.vg_dir
	n_works = args.n_works
	metadata_name = args.metadata_name
	plan_name = args.plan_name
	name = args.name
	force = args.force

	metadata_path = os.path.join(f'{vg_dir}', f'metadata-{metadata_name}.json')
	qas_path = os.path.join(f'{vg_dir}', f'qa-plans-{plan_name}.json')
	instruction_path = os.path.join(f'{vg_dir}', f'{name}.json')

	sg_dataset = JointDataset(
		f"{vg_dir}/scene-graph-annotations.json",
		depth_dir_path=f"{vg_dir}/depth",
		seg_dir_path=f"{vg_dir}/segment" # _pkl
	)

	generator_list = (
			# ObjectGeneratorList 
			# +
			AttributeGeneratorList 
			# + RelationGeneratorList
			# + ObjectDepthGenerator
	) # type: ignore

	gen = JointGenerator(
		dataset=sg_dataset,
		generators=generator_list,
		template_mode='qa',
		return_templated=False,
		n_sample_per_generator=1,
		data_version=name
	)

	if os.path.exists(metadata_path) and not force:
		metadata = json.load(open(metadata_path))
	else:
		metadata = gen.collect_metadata(n_works)
		json.dump(metadata, open(metadata_path, 'w'), indent=4)

	if os.path.exists(qas_path) and not force:
		qas = json.load(open(qas_path))
	else:
		qas = gen.generate(metadata_count=metadata, n_workers=n_works)

	metadata_count = {}
	for i in qas:
		for k, vs in i['metadata'].items():
			if k not in metadata_count:
				metadata_count[k] = {}
			for v in vs:
				if v not in metadata_count[k]:
					metadata_count[k][v] = 0
				metadata_count[k][v] += 1
	# pprint(metadata_count)
	print(len(qas))
	json.dump(qas, open(qas_path, 'w'), indent=4)

	instructions = gen.tool_template(qas, multiple_choice_ratio=0.0)
	mc_instructions = gen.tool_template(qas, multiple_choice_ratio=1.0)

	instruction_path = os.path.join(f'{vg_dir}', f'{name}.json')
	json.dump(instructions, open(instruction_path, 'w'), indent=4)

	mc_name = f"{name}_mc"
	mc_instruction_path = os.path.join(f'{vg_dir}', f'{mc_name}.json')
	json.dump(mc_instructions, open(mc_instruction_path, 'w'), indent=4)