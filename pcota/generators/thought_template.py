def get_thought_template(tool_name):
	return eval(f"{tool_name}_template()")

def GetAttributesOfObject_template():
    thought_templates = ["I need to get the attributes of {object} in the {image_kw}.", 
                         "I need to analyze the {object}'s attributes in the {image_kw}.",
                         "To answer the question, I need to identify the attributes of {object}.",
                         "I need to determine the attributes of {object} to answer the question.", 
                         "I should analyze the properties of the {object}."]
    return thought_templates

def GetRelationshipsBetweenObjects_template():
    thought_templates = ["I need to determine the relationships between {object1} and {object2} in the {image_kw}.",
                        "I need to analyze the relationships between {object1} and {object2} in the {image_kw}.",
                        "To answer the question involving {object1} and {object2}, I need to analyze the relationships between them.",
                        "I need to list the relationships between {object1} and {object2} to answer the question.",
                        "I should determine the connections between {object1} and {object2}."]
    return thought_templates

def GetObjects_template():
    thought_templates = ["I need to check what objects are present in the {image_kw}.", 
                        "I need to analyze the {image_kw} for context."
                        "I need to identify the objects in the {image_kw}.",
                        "To answer the question, let's first analyze the {image_kw}.",
                        "To answer the question, analyzing the objects in the {image_kw} is necessary."]
    return thought_templates

def LocalizeObjects_template():
    thought_templates = ["I need to analyze the positions of {objects} in the {image_kw}.", 
                        "I need to analyze the locations of {objects} in the {image_kw}.", 
                        "I need to localize the {objects} based on the {image_kw}.",
                        "I'll identify the positions of {objects} in the {image_kw}.",
                        "I need to determine the positions of {objects} by analyzing the {image_kw}."]
    return thought_templates

def EstimateObjectDepth_template():
    thought_templates = ["I should estimate the depth of {object} to determine whether it is closer or farther.", 
                         "I will estimate the depth of {object}.", 
                         "I need to estimate the depth for {object} to make a comparison.", 
                         "To determine how far {object} is, I need to evaluate the distance to it.",
                         "I now need to estimate the depth for {object}."]
    return thought_templates


def EstimateRegionDepth_template():
    thought_templates = ["I should estimate the objects' depths to determine which one is closer.", 
                         "I need to estimate the region's depth in the image.", 
                         "I need to determine the depths of the detected objects based on their positions.",
                         "I need to estimate the depth of the objects to make a comparison.",
                         "To determine the relative proximity of the objects in the image, I need to estimate the depth of each object."]
    return thought_templates

def Terminate_template():
    thought_templates = ["Based on the information above, I can conclude that the answer is {answer}",
                         "Based on a close analysis of the {image_kw} and additional information above, I believe the answer is {answer}.",
                         "I have analyzed the {image_kw} and the information above, and I believe the answer is {answer}.",
                         "The {image_kw} and the information above suggest that the answer is {answer}.", 
                         "According to the content of the {image_kw} and the observations, I can conclude that the answer is {answer}."]
    return thought_templates