import argparse
from oneibl.one import ONE


parser = argparse.ArgumentParser(description='Load in subject info')

parser.add_argument('-s', '--subject', default=False, required=True, help='Subject Name')
parser.add_argument('-d', '--date', default=False, required=True,
                    help='Date of session YYYY-MM-DD')
parser.add_argument('-n', '--session_no', default=False, required=True, help='Session Number')
parser.add_argument('-e', '--ephys_data', default=False, required=False,
                    help='Set True to load ephys.bin data')
args = parser.parse_args()

one = ONE()
eid = one.search(subject=str(args.subject), date=str(args.date), number=args.session_no)[0]
data_path = one.load(eid, clobber=False, download_only=True)
print(f'Successfully loaded data for {args.subject} {args.date} session no. {args.session_no}')

if args.ephys_data:
    print('Loading ephys.bin data')
    ephys_types = ['ephysData.raw.ch', 'ephysData.raw.meta', 'ephysData.raw.ap', 'ephysData.raw.lf']
    ephys_path = one.load(eid, dataset_types=ephys_types, clobber=False, download_only=True)
else:
    print('Ephys flag set to false... will not load ephys data')
