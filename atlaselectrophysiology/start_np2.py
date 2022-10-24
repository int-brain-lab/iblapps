# SH012/2020-01-31/001
# todo: cache the destriping somewhere
# todo: propagate and split-4 buttons
from one.api import ONE
from iblapps.atlaselectrophysiology.ephys_atlas_gui import viewer, MainWindow


# 141bf1d0-ae0d-4d13-802a-fc61e7aa98ee probe00c {'subject': 'HB_004', 'start_time': '2022-09-02T15:21:21.646371', 'number': 10, 'lab': 'steinmetzlab', 'id': 'edce7eab-4f75-4c3a-8cd9-a797ec1bd45b', 'task_protocol': '_iblrig_tasks_ephysChoiceWorld6.6.2'}
# ba510211-670b-49b6-ae65-795733cf9fbe probe00d {'subject': 'HB_004', 'start_time': '2022-09-02T15:21:21.646371', 'number': 10, 'lab': 'steinmetzlab', 'id': 'edce7eab-4f75-4c3a-8cd9-a797ec1bd45b', 'task_protocol': '_iblrig_tasks_ephysChoiceWorld6.6.2'}
# d86b0b7b-bb61-4336-b18d-b018a560f67f probe00a {'subject': 'HB_004', 'start_time': '2022-09-02T15:21:21.646371', 'number': 10, 'lab': 'steinmetzlab', 'id': 'edce7eab-4f75-4c3a-8cd9-a797ec1bd45b', 'task_protocol': '_iblrig_tasks_ephysChoiceWorld6.6.2'}
# f8bb09c8-4fb6-414e-9644-a1640b894c31 probe00b {'subject': 'HB_004', 'start_time': '2022-09-02T15:21:21.646371', 'number': 10, 'lab': 'steinmetzlab', 'id': 'edce7eab-4f75-4c3a-8cd9-a797ec1bd45b', 'task_protocol': '_iblrig_tasks_ephysChoiceWorld6.6.2'}

# 5069d734-1ef9-419b-ab9b-9e7ffc22ec15 probe00 {'subject': 'NR_0020', 'start_time': '2022-05-12T16:10:05.336387', 'number': 1, 'lab': 'steinmetzlab', 'id': 'aed404ce-b3fb-454b-ac43-2f12198c9eaf', 'task_protocol': '_iblrig_tasks_ephysChoiceWorld6.5.3'}
# d21acc7a-29ae-4122-847a-6017e9243382 probe01 {'subject': 'NR_0020', 'start_time': '2022-05-12T16:10:05.336387', 'number': 1, 'lab': 'steinmetzlab', 'id': 'aed404ce-b3fb-454b-ac43-2f12198c9eaf', 'task_protocol': '_iblrig_tasks_ephysChoiceWorld6.5.3'}
one = ONE(base_url='https://alyx.internationalbrainlab.org')
eid, pname = ('edce7eab-4f75-4c3a-8cd9-a797ec1bd45b', 'probe00')
shanks = one.alyx.rest('insertions', 'list', session=eid, django=f"name__istartswith,{pname}")

guis = {}
for sh in shanks:
    guis[sh['name']] = viewer(probe_id=sh['id'], one=None, histology=True, title=f'{sh["name"]}')

assert len(set([id(g.slice_data) for g in guis.values()])) == 1
assert len(set([id(g.loaddata.brain_atlas) for g in guis.values()])) == 1

MainWindow.explode()
