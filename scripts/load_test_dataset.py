# from datasets import load_dataset
#
# dataset_paths = [
#     ['ceval/ceval-exam',
#      ['computer_network', 'operating_system', 'computer_architecture', 'college_programming', 'college_physics',
#       'college_chemistry', 'advanced_mathematics', 'probability_and_statistics', 'discrete_mathematics',
#       'electrical_engineer', 'metrology_engineer', 'high_school_mathematics', 'high_school_physics',
#       'high_school_chemistry', 'high_school_biology', 'middle_school_mathematics', 'middle_school_biology',
#       'middle_school_physics', 'middle_school_chemistry', 'veterinary_medicine', 'college_economics',
#       'business_administration', 'marxism', 'mao_zedong_thought', 'education_science', 'teacher_qualification',
#       'high_school_politics', 'high_school_geography', 'middle_school_politics', 'middle_school_geography',
#       'modern_chinese_history', 'ideological_and_moral_cultivation', 'logic', 'law', 'chinese_language_and_literature',
#       'art_studies', 'professional_tour_guide', 'legal_professional', 'high_school_chinese', 'high_school_history',
#       'middle_school_history', 'civil_servant', 'sports_science', 'plant_protection', 'basic_medicine',
#       'clinical_medicine', 'urban_and_rural_planner', 'accountant', 'fire_engineer',
#       'environmental_impact_assessment_engineer', 'tax_accountant', 'physician']],  # ceval*
#     ['haonan-li/cmmlu', [
#         'agronomy', 'anatomy', 'ancient_chinese', 'arts', 'astronomy', 'business_ethics',
#         'chinese_civil_service_exam', 'chinese_driving_rule', 'chinese_food_culture',
#         'chinese_foreign_policy', 'chinese_history', 'chinese_literature',
#         'chinese_teacher_qualification', 'clinical_knowledge', 'college_actuarial_science',
#         'college_education', 'college_engineering_hydrology', 'college_law',
#         'college_mathematics', 'college_medical_statistics', 'college_medicine',
#         'computer_science', 'computer_security', 'conceptual_physics',
#         'construction_project_management', 'economics', 'education', 'electrical_engineering',
#         'elementary_chinese', 'elementary_commonsense', 'elementary_information_and_technology',
#         'elementary_mathematics', 'ethnology', 'food_science', 'genetics', 'global_facts',
#         'high_school_biology', 'high_school_chemistry', 'high_school_geography',
#         'high_school_mathematics', 'high_school_physics', 'high_school_politics',
#         'human_sexuality', 'international_law', 'journalism', 'jurisprudence',
#         'legal_and_moral_basis', 'logical', 'machine_learning', 'management', 'marketing',
#         'marxist_theory', 'modern_chinese', 'nutrition', 'philosophy', 'professional_accounting',
#         'professional_law', 'professional_medicine', 'professional_psychology',
#         'public_relations', 'security_study', 'sociology', 'sports_science',
#         'traditional_chinese_medicine', 'virology', 'world_history', 'world_religions'
#     ]],  # cmmlu*
#     ['tyouisen/aclue',
#      ['polysemy_resolution', 'poetry_sentiment_analysis', 'named_entity_recognition', 'basic_ancient_chinese',
#       'poetry_context_prediction', 'sentence_segmentation', 'couplet_prediction', 'poetry_appreciate',
#       'ancient_chinese_culture', 'ancient_phonetics', 'homographic_character_resolution', 'ancient_literature',
#       'ancient_medical', 'poetry_quality_assessment', 'reading_comprehension']],  # aclue
#     ['juletxara/mgsm', ['zh']],  # mgsm_direct_zh
#     ['openbookqa', ['main']],  # openbookqa
#     ['ZoneTwelve/tmmluplus',
#      ['dentistry', 'traditional_chinese_medicine_clinical_medicine', 'clinical_psychology', 'technical',
#       'culinary_skills', 'mechanical', 'logic_reasoning', 'real_estate', 'general_principles_of_law', 'finance_banking',
#       'anti_money_laundering', 'ttqav2', 'marketing_management', 'business_management', 'organic_chemistry',
#       'advance_chemistry', 'physics', 'secondary_physics', 'human_behavior', 'national_protection', 'jce_humanities',
#       'politic_science', 'agriculture', 'official_document_management', 'financial_analysis', 'pharmacy',
#       'educational_psychology', 'statistics_and_machine_learning', 'management_accounting', 'introduction_to_law',
#       'computer_science', 'veterinary_pathology', 'accounting', 'fire_science', 'optometry', 'insurance_studies',
#       'pharmacology', 'taxation', 'education_(profession_level)', 'economics', 'veterinary_pharmacology',
#       'nautical_science', 'occupational_therapy_for_psychological_disorders', 'trust_practice', 'geography_of_taiwan',
#       'physical_education', 'auditing', 'administrative_law', 'basic_medical_science', 'macroeconomics', 'trade',
#       'chinese_language_and_literature', 'tve_design', 'junior_science_exam', 'junior_math_exam', 'junior_chinese_exam',
#       'junior_social_studies', 'tve_mathematics', 'tve_chinese_language', 'tve_natural_sciences', 'junior_chemistry',
#       'music', 'education', 'three_principles_of_people', 'taiwanese_hokkien', 'engineering_math', 'linear_algebra']]
#     # tmmluplus
#
# ]
#
# for dataset_path in dataset_paths:
#     for dataset_name in dataset_path[1]:
#         datasets = load_dataset(dataset_path[0], dataset_name, cache_dir='./test_dataset_cache')
#
# """
# export HF_HUB_OFFLINE=1 && lm_eval --model hf --model_args pretrained=/xxx/minimind/minimind-v2-small/,device=cuda,dtype=auto --tasks ceval* --batch_size 8 --trust_remote_code
# """
"""
$env:HF_HUB_OFFLINE=1; lm_eval --model hf --model_args pretrained=../minimind-v2-small/,device=cuda,dtype=auto --tasks ceval* --batch_size 8 --trust_remote_code
"""

import subprocess

# 定义要执行的命令
command = (
    'set HF_HUB_OFFLINE=1 & '
    'lm_eval --model hf --model_args pretrained=../minimind-v2-small/,device=cuda,dtype=auto '
    '--tasks ceval* --batch_size 8 --trust_remote_code'
)

# 使用 subprocess 执行命令
try:
    process = subprocess.run(
        command,
        shell=True,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # 打印命令的输出
    print("STDOUT:", process.stdout)
    print("STDERR:", process.stderr)
except subprocess.CalledProcessError as e:
    print(f"命令执行失败，返回码: {e.returncode}")
    print("STDERR:", e.stderr)