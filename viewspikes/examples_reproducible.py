from oneibl.one import ONE
from atlaselectrophysiology.alignment_with_easyqc import viewer

one = ONE()


pids = ['ce397420-3cd2-4a55-8fd1-5e28321981f4',
       'e31b4e39-e350-47a9-aca4-72496d99ff2a',
       'f8d0ecdc-b7bd-44cc-b887-3d544e24e561',
       '6fc4d73c-2071-43ec-a756-c6c6d8322c8b',
       'c17772a9-21b5-49df-ab31-3017addea12e',
       '0851db85-2889-4070-ac18-a40e8ebd96ba',
       'eeb27b45-5b85-4e5c-b6ff-f639ca5687de',
       '69f42a9c-095d-4a25-bca8-61a9869871d3',
       'f03b61b4-6b13-479d-940f-d1608eb275cc',
       'f2ee886d-5b9c-4d06-a9be-ee7ae8381114',
       'f26a6ab1-7e37-4f8d-bb50-295c056e1062',
       'c4f6665f-8be5-476b-a6e8-d81eeae9279d',
       '9117969a-3f0d-478b-ad75-98263e3bfacf',
       'febb430e-2d50-4f83-87a0-b5ffbb9a4943',
       '8413c5c6-b42b-4ec6-b751-881a54413628',
       '8b7c808f-763b-44c8-b273-63c6afbc6aae',
       'f936a701-5f8a-4aa1-b7a9-9f8b5b69bc7c',
       '63517fd4-ece1-49eb-9259-371dc30b1dd6',
       '8d59da25-3a9c-44be-8b1a-e27cdd39ca34',
       '19baa84c-22a5-4589-9cbd-c23f111c054c',
       '143dd7cf-6a47-47a1-906d-927ad7fe9117',
       '84bb830f-b9ff-4e6b-9296-f458fb41d160',
       'b749446c-18e3-4987-820a-50649ab0f826',
       '36362f75-96d8-4ed4-a728-5e72284d0995',
       '9657af01-50bd-4120-8303-416ad9e24a51',
       'dab512bd-a02d-4c1f-8dbc-9155a163efc0']



INDEX = 22
pid = pids[INDEX]
av = viewer(pid, one=one)



