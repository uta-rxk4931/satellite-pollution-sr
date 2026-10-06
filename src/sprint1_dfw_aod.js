var dfw = ee.Geometry.Rectangle([-97.8, 32.3, -96.3, 33.5]);

var img = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES')
  .filterDate('2023-06-15', '2023-06-16')
  .filterBounds(dfw)
  .select(['Optical_Depth_055', 'AOD_QA'])
  .first();

var aod = img.select('Optical_Depth_055').multiply(0.001);
var qa = img.select('AOD_QA');
var masked = aod.updateMask(qa.gte(0)).clip(dfw);

Map.centerObject(dfw, 8);
Map.addLayer(masked, {min: 0, max: 0.6, palette: ['white', 'yellow', 'red']}, 'AOD 550');

var stats = masked.reduceRegion({
  reducer: ee.Reducer.min().combine(ee.Reducer.median(), '', true)
      .combine(ee.Reducer.max(), '', true),
  geometry: dfw,
  scale: 1000,
  maxPixels: 1e8
});
print('DFW AOD min / median / max', stats);
print('If median is above 5, the 0.001 scale was not applied.');