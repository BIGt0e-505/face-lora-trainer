"""
Regularisation Prompt Templates
================================
Varied prompts for generating regularisation images.
Each prompt includes {class_word} which gets replaced with the actual class word (e.g., "man", "woman", "person").

The variety helps prevent the LoRA from associating specific styles, poses, or contexts with the class word.
"""

# Prompt templates - {class_word} will be replaced with actual class word
REG_PROMPT_TEMPLATES = [
    # Basic variations
    "a {class_word}",
    "a {class_word} looking at the camera",
    "a {class_word} facing forward",
    "a {class_word} with a neutral expression",
    
    # Angles and poses
    "a {class_word} in profile view",
    "a {class_word} from the side",
    "a {class_word} looking to the left",
    "a {class_word} looking to the right",
    "a {class_word} looking up",
    "a {class_word} looking down",
    "a {class_word} in three quarter view",
    "a {class_word} turned slightly",
    "a {class_word} head tilted",
    "a {class_word} looking over shoulder",
    
    # Expressions
    "a {class_word} smiling",
    "a {class_word} with a slight smile",
    "a {class_word} laughing",
    "a {class_word} with a serious expression",
    "a {class_word} looking thoughtful",
    "a {class_word} looking relaxed",
    "a {class_word} looking confident",
    "a {class_word} with a calm expression",
    
    # Lighting variations
    "a {class_word} in natural light",
    "a {class_word} in soft lighting",
    "a {class_word} in warm lighting",
    "a {class_word} in cool lighting",
    "a {class_word} with dramatic lighting",
    "a {class_word} in golden hour light",
    "a {class_word} in diffused light",
    "a {class_word} with side lighting",
    "a {class_word} in bright daylight",
    "a {class_word} in shade",
    "a {class_word} backlit",
    "a {class_word} in studio lighting",
    
    # Indoor locations
    "a {class_word} indoors",
    "a {class_word} in an office",
    "a {class_word} in a living room",
    "a {class_word} at home",
    "a {class_word} in a cafe",
    "a {class_word} in a restaurant",
    "a {class_word} in a library",
    "a {class_word} at a desk",
    "a {class_word} by a window",
    "a {class_word} in a kitchen",
    
    # Outdoor locations
    "a {class_word} outdoors",
    "a {class_word} outside",
    "a {class_word} in a park",
    "a {class_word} in a garden",
    "a {class_word} on a street",
    "a {class_word} in a city",
    "a {class_word} in nature",
    "a {class_word} at the beach",
    "a {class_word} in a forest",
    "a {class_word} in the mountains",
    "a {class_word} by water",
    
    # Backgrounds
    "a {class_word} with a plain background",
    "a {class_word} with a blurred background",
    "a {class_word} against a wall",
    "a {class_word} with a neutral background",
    "a {class_word} with greenery behind",
    "a {class_word} with urban background",
    
    # Framing
    "close up of a {class_word}",
    "a {class_word} headshot",
    "upper body shot of a {class_word}",
    "a {class_word} from the shoulders up",
    "a {class_word} from the chest up",
    "medium shot of a {class_word}",
    "a {class_word} half body",
    
    # Clothing (generic)
    "a {class_word} in casual clothes",
    "a {class_word} wearing a shirt",
    "a {class_word} wearing a t-shirt",
    "a {class_word} in a jacket",
    "a {class_word} in a sweater",
    "a {class_word} in formal attire",
    "a {class_word} in business casual",
    "a {class_word} wearing a hoodie",
    "a {class_word} in a coat",
    
    # Activities (subtle)
    "a {class_word} standing",
    "a {class_word} sitting",
    "a {class_word} leaning",
    "a {class_word} relaxing",
    "a {class_word} waiting",
    "a {class_word} thinking",
    "a {class_word} resting",
    
    # Time of day
    "a {class_word} in morning light",
    "a {class_word} in afternoon light",
    "a {class_word} in evening light",
    "a {class_word} at dusk",
    
    # Weather/atmosphere
    "a {class_word} on a sunny day",
    "a {class_word} on a cloudy day",
    "a {class_word} on an overcast day",
    
    # Combinations
    "a {class_word} smiling in natural light",
    "a {class_word} looking relaxed outdoors",
    "a {class_word} with a neutral expression indoors",
    "a {class_word} in casual clothes looking at camera",
    "a {class_word} sitting by a window",
    "a {class_word} standing in a park",
    "a {class_word} in profile with soft lighting",
    "a {class_word} headshot with plain background",
    "a {class_word} in three quarter view smiling",
    "a {class_word} outdoors in natural light",
    "a {class_word} indoors with warm lighting",
    "a {class_word} looking thoughtful by window",
    "a {class_word} in casual attire relaxing",
    "a {class_word} with confident expression",
    "a {class_word} in urban setting",
    "a {class_word} close up with neutral background",
    "a {class_word} medium shot outdoors",
    "a {class_word} in golden hour lighting smiling",
    "a {class_word} looking to the side thoughtfully",
    "a {class_word} in a cafe looking relaxed",
    "a {class_word} at desk looking at camera",
    "a {class_word} in garden with natural light",
    "a {class_word} wearing jacket outdoors",
    "a {class_word} in sweater indoors",
    "a {class_word} headshot with soft lighting",
    "a {class_word} upper body in office",
    "a {class_word} sitting relaxed",
    "a {class_word} standing confidently",
    "a {class_word} leaning against wall",
    "a {class_word} in shade looking calm",
    "a {class_word} backlit with warm tones",
    "a {class_word} in cool lighting looking serious",
    "a {class_word} laughing in bright daylight",
    "a {class_word} with slight smile in studio",
    "a {class_word} in nature looking peaceful",
    "a {class_word} on street in city",
    "a {class_word} by water in evening",
    "a {class_word} in morning light smiling",
    "a {class_word} afternoon relaxed expression",
    "a {class_word} wearing t-shirt casual",
    "a {class_word} in formal attire confident",
    "a {class_word} hoodie casual indoors",
    "a {class_word} looking over shoulder smiling",
    "a {class_word} head tilted thoughtful",
    "a {class_word} three quarter dramatic lighting",
    "a {class_word} profile in golden hour",
    "a {class_word} facing forward neutral background",
    "a {class_word} close up natural expression",
    "a {class_word} medium shot urban background",
    "a {class_word} half body in park",
    "a {class_word} shoulders up soft light",
    "a {class_word} chest up by window",
    "a {class_word} in library reading light",
    "a {class_word} restaurant ambient lighting",
    "a {class_word} living room relaxed",
    "a {class_word} kitchen casual",
    "a {class_word} forest dappled light",
    "a {class_word} beach natural light",
    "a {class_word} mountains scenic background",
    "a {class_word} sunny day happy",
    "a {class_word} cloudy soft diffused",
    "a {class_word} overcast even lighting",
    "a {class_word} dusk warm colors",
    "a {class_word} plain white background",
    "a {class_word} blurred bokeh background",
    "a {class_word} greenery natural setting",
    "a {class_word} brick wall urban",
    "a {class_word} looking up hopeful",
    "a {class_word} looking down contemplative",
    "a {class_word} eyes closed peaceful",
    "a {class_word} slight squint sunny",
    "a {class_word} wide smile genuine",
    "a {class_word} subtle smile warm",
    "a {class_word} serious focused",
    "a {class_word} calm serene",
    "a {class_word} energetic bright",
    "a {class_word} tired relaxed",
    "a {class_word} alert attentive",
    "a {class_word} distracted looking away",
    "a {class_word} engaged looking at camera",
    "a {class_word} casual stance",
    "a {class_word} formal posture",
    "a {class_word} relaxed posture",
    "a {class_word} arms crossed",
    "a {class_word} hands in pockets",
    "a {class_word} hand on chin",
    "a {class_word} touching hair",
    "a {class_word} adjusting glasses",
    "a {class_word} holding coffee",
    "a {class_word} with phone",
]


def get_reg_prompts(class_word: str, num_prompts: int = 200) -> list[str]:
    """
    Get a list of varied regularisation prompts with the class word inserted.
    
    Args:
        class_word: The class word to insert (e.g., "man", "woman", "person")
        num_prompts: Number of prompts to return
    
    Returns:
        List of prompt strings with {class_word} replaced
    """
    prompts = [template.format(class_word=class_word) for template in REG_PROMPT_TEMPLATES]
    
    # If we need more prompts than templates, cycle through them
    while len(prompts) < num_prompts:
        prompts.extend(prompts)
    
    return prompts[:num_prompts]
