# Sample Images

This directory contains sample images from the COCO 2014 dataset for demonstration purposes.

## Adding Sample Images

1. Download sample images from COCO 2014:
   - Visit: https://cocodataset.org/#download
   - Download: 2014 Val images (6GB)

2. Select diverse images:
   - People and objects
   - Indoor and outdoor scenes
   - Simple and complex compositions
   - Various lighting conditions

3. Place 10-20 sample images here with descriptive names:
   ```
   samples/
   ├── beach_people_001.jpg
   ├── city_street_002.jpg
   ├── living_room_003.jpg
   ├── sports_soccer_004.jpg
   ├── animals_dog_005.jpg
   └── ...
   ```

## Image Requirements

- **Format**: JPG, JPEG, or PNG
- **Size**: Ideally 640x480 or larger
- **File size**: Under 5MB per image
- **Quality**: Good lighting, clear subjects

## Sample Categories to Include

1. **People**: Single person, groups, activities
2. **Animals**: Dogs, cats, birds, wildlife
3. **Vehicles**: Cars, buses, bicycles
4. **Indoor**: Kitchen, living room, bedroom
5. **Outdoor**: Street, park, beach, mountain
6. **Food**: Meals, fruits, prepared dishes
7. **Sports**: Various sports activities
8. **Objects**: Furniture, electronics, everyday items

## Automatic Sample Manifest

The app will automatically generate a manifest from images in this directory using `create_sample_manifest()` in `utils/coco_utils.py`.

## Example File Names

Good naming convention:
- `person_bicycle_street_001.jpg`
- `dog_frisbee_park_002.jpg`
- `kitchen_dining_food_003.jpg`

This helps users quickly identify scene content.

---

**Note**: This directory is tracked in git but large image files should be added to .gitignore if needed.
