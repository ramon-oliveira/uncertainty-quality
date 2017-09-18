import sys
import os
import json
from mako.template import Template
from mako import exceptions


template_file = os.path.abspath(sys.argv[1])
run_id = sys.argv[2]

config = json.load(open('runs/{}/config.json'.format(run_id)))
run = json.load(open('runs/{}/run.json'.format(run_id)))
info = json.load(open('runs/{}/info.json'.format(run_id)))

template = Template(filename=template_file)

try:
    out = template.render(config=config, run=run, info=info)
    open('runs/{}/report.html'.format(run_id), 'w').write(out)
    print('success!')
except:
    print('fail!')
    mako_exception = exceptions.html_error_template().render().decode('utf-8')
    open('runs/{}/exception.html'.format(run_id), 'w').write(mako_exception)
