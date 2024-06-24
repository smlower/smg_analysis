```
NOTE an alternative way to get this is todo:

m = ModelOutput('example.050.rtout.sed')
data = m.get_quantities()
data.quantities['temperature']

will give a 3D array, which should be weighted by the VSG/USG etc. mass contributions

```



from hyperion.model import ModelOutput
import yt

run = '/Users/desika/pd_runs/mufasa_zooms/m50n512/z2/halo18_ml11/smg_survey/example.080.rtout.sed'

m = ModelOutput(run)
oct = m.get_quantities()
pf = oct.to_yt()
ad = pf.all_data()
print pf.derived_field_list

tdust = ad[ ('gas', 'temperature')]
