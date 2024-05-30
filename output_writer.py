import json
import os


class OutputWriter:
    def __init__(self, output_path):
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def write_output(self, data, file_name, indent=None):
        with open('{}/{}.json'.format(self.output_path, file_name), 'w') as outfile:
            json.dump(data, outfile, indent=indent)
