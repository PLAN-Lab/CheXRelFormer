import pandas as pd
import os

def main():
    output_dir = '~/CXR/data/chest_imagenome/1.0.0/gk_parser_code'
    comp_df = pd.read_csv('~/CXR/data/chest_imagenome/1.0.0/silver_dataset/scene_tabular/comparison_relations_tabular.txt', sep='\t')
    #The comparison_relations_tabular has the following columns: relationship_id,subject_id,object_id,bbox,comparison,attribute,sentence,bbox_coord_224_subject,bbox_coord_224_object,bbox_coord_original_subject,bbox_coord_original_object
    #The file is generated from chest imagenome dataset
    bbox_set = set(['right lung', 'right apical zone', 'right upper lung zone', 'right mid lung zone', \
            'right lower lung zone', 'right hilar structures', 'right costophrenic angle', \
            'left lung', 'left apical zone', 'left upper lung zone', 'left mid lung zone', \
            'left lower lung zone', 'left hilar structures', 'left costophrenic angle', 'mediastinum', \
            'upper mediastinum', 'cardiac silhouette', 'trachea'])
    comparison_set = set(['improved', 'worsened', 'no change'])
    comp_df = comp_df[comp_df['comparison'].isin(comparison_set) & comp_df['bbox'].isin(bbox_set)]
    comp_df['current_image_id'] = comp_df.apply(lambda x: x.subject_id.split('_')[0], axis=1)
    comp_df['previous_image_id'] = comp_df.apply(lambda x: x.object_id.split('_')[0], axis=1)
    comp_df['label_name'] = comp_df.apply(lambda x: x.attribute.split('|')[2], axis=1)
    comp_df['category_ID'] = comp_df.apply(lambda x: x.attribute.split('|')[0], axis=1)
    comp_df = comp_df[comp_df['category_ID'].isin(['anatomicalfinding', 'disease'])]
    comp_df.to_csv(os.path.join(output_dir,'final_comparison_dataset.txt')
                         ,sep='\t',encoding='utf-8',index=False)

#The generated final_comparison_dataset.txt has the following columns: relationship_id,	subject_id,	object_id,	bbox,	comparison,	attribute,	sentence,	bbox_coord_224_subject,	bbox_coord_224_object,	bbox_coord_original_subject,	bbox_coord_original_object,	current_image_id,	previous_image_id,	label_name,	category_ID

if __name__ == 'main':
    main()