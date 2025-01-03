import argparse
import os
from pprint import pprint

from pcota.base import *
from pcota.generators.tool.aggregate import *
from pcota.generators.tool.compare import *
from pcota.generators.tool.select import *

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--vg_dir', type=str, default="sample_data/")
	parser.add_argument('--n_data', type=int, default=2)
	parser.add_argument('--metadata_name', type=str, default="v3-multi-tool")
	parser.add_argument('--name', type=str, default="v3-multi-tool")
	parser.add_argument('--force', action="store_true", default=False)
	args = parser.parse_args()

	vg_dir = args.vg_dir
	metadata_name = f"{args.metadata_name}-{args.n_data}"
	name = f"{args.name}-{args.n_data}"
	force = args.force

	metadata_path = os.path.join(f'{vg_dir}', f'metadata-{metadata_name}.json')
	qas_path = os.path.join(f'{vg_dir}', f'qa-plans-{metadata_name}.json')

	sg_dataset = JointDataset(
		f"{vg_dir}/scene-graph-annotations.json",
		depth_dir_path=f"{vg_dir}/depth",
		seg_dir_path=f"{vg_dir}/segment"
	)

	generator_list = (
			# MultiSelectGeneratorList
			# +
			# MultiAggregateGeneratorList
			# +
			MultiCompareGeneratorList
	)

	gen = JointGenerator(
		dataset=sg_dataset,
		generators=generator_list,
		template_mode='qa',
		return_templated=False,
		n_data=args.n_data,
		n_sample=100,
		data_version=name
	)

	if os.path.exists(qas_path) and not force:
		qas = json.load(open(qas_path))
	else:
		qas = gen.generate()

		metadata_count = {}
		for i in qas:
			for k, vs in i['metadata'].items():
				if k not in metadata_count:
					metadata_count[k] = {}
				for v in vs:
					if v not in metadata_count[k]:
						metadata_count[k][v] = 0
					metadata_count[k][v] += 1
		pprint(metadata_count)

		json.dump(qas, open(qas_path, 'w'), indent=4)

	instructions = gen.multi_image_tool_template(qas, multiple_choice_ratio=0.0)
	instruction_path = os.path.join(f'{vg_dir}', f'{name}.json')
	json.dump(instructions, open(instruction_path, 'w'), indent=4)

	mc_instructions = gen.multi_image_tool_template(qas, multiple_choice_ratio=1.0)
	mc_name = f"{name}_mc"
	mc_instruction_path = os.path.join(f'{vg_dir}', f'{mc_name}.json')
	json.dump(mc_instructions, open(mc_instruction_path, 'w'), indent=4)
