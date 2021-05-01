class API:

    def __init__(self, package_name, class_name, class_description):
        self.package_name = package_name
        self.class_name = class_name
        self.class_description = class_description
        self.methods = []
        self.methods_descriptions_pure_text = []
        self.methods_descriptions = []  #the description is segmented into words
        self.methods_descriptions_stemmed = []
        self.methods_matrix = []
        self.methods_idf_vector = []
        self.class_description_matrix = None
        self.class_description_idf_vector = None


    def print_api(self):
        print(self.package_name+'.'+self.class_name,self.class_description)


class Question:

    def __init__(self,id,title,body,score,view_count,accepted_answer_id):
        self.id = id
        self.title = title
        self.body = body
        self.accepted_answer_id = accepted_answer_id
        self.score = score
        self.view_count = view_count
        self.answers = list()
        self.title_words = None
        self.matrix = None
        self.idf_vector = None

class Answer:

    def __init__(self, id, parent_id, body, score):
        self.id = id
        self.parent_id = parent_id
        self.body = body
        self.score = score



class Record:
    def __init__(self,title,method_name,method_block_flat,method_api_sequence,decompose_methodname):
        self.title=title
        self.title_words=None
        self.title_matrix=None
        self.title_idf_vector=None
        self.method_name=method_name
        self.method_block_flat=method_block_flat
        self.method_api_sequence=method_api_sequence
        self.decompose_methodname=decompose_methodname
        self.full_title=title+" "+decompose_methodname
        self.full_title_words = None
        self.full_title_matrix = None
        self.full_title_idf_vector = None

class Experiment:
    def __init__(self, method_annotation, now_method_flat, true_api,now_api):
        self.method_annotation=method_annotation
        self.now_method_flat=now_method_flat
        self.true_api=true_api
        self.now_api=now_api