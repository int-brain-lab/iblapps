import argparse
from one.api import ONE
import numpy as np

parser = argparse.ArgumentParser(description='Load in subject info')

parser.add_argument('-s', '--subject', default=False, required=False,
                    help='Subject Name')
parser.add_argument('-d', '--date', default=False, required=False,
                    help='Date of session YYYY-MM-DD')
parser.add_argument('-n', '--session_no', default=1, required=False,
                    help='Session Number', type=int)
parser.add_argument('-e', '--eid', default=False, required=False,
                    help='Session eid')
parser.add_argument('-p', '--probe_label', default=False, required=True,
                    help='Probe Label')

args = parser.parse_args()

one = ONE()
if not args.eid:
    if not np.all(np.array([args.subject, args.date, args.session_no],
                           dtype=object)):
        print('Must give Subject, Date and Session number')
    else:
        eid = one.search(subject=str(args.subject), date=str(args.date), number=args.session_no)[0]
        print(eid)
else:
    eid = str(args.eid)

_ = one.load_object(eid, obj='trials', collection='alf',
                    download_only=True)
_ = one.load_object(eid, obj='spikes', attribute='times|clusters|amps|depths',
                    collection=f'alf/{str(args.probe_label)}', download_only=True)
_ = one.load_object(eid, obj='clusters', collection=f'alf/{str(args.probe_label)}',
                    download_only=True)
