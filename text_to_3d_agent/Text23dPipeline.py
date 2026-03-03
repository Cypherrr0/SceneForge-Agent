import os ,io
import gc
import sys
import base64
import torch
import shutil
from PIL import Image
from pathlib import Path
# from google import genai
from openai import OpenAI
from typing import  Dict, Any,Union
# from dotenv import load_dotenv
# from google.genai import types

# 获取项目根目录（Text23dPipeline.py 在 text_to_3d_agent/ 下）
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 使用绝对路径
sys.path.insert(0, str(PROJECT_ROOT / 'text_to_3d_agent' / 'DiffSynth-Studio'))
sys.path.insert(0, str(PROJECT_ROOT / 'text_to_3d_agent' / 'Hunyuan3D-2.1' / 'hy3dshape'))
sys.path.insert(0, str(PROJECT_ROOT / 'text_to_3d_agent' / 'Hunyuan3D-2.1' / 'hy3dpaint'))
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline,export_to_trimesh
from hy3dshape.rembg import BackgroundRemover
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from convert_utils import create_glb_with_pbr_materials
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

def quick_convert_with_obj2gltf(obj_path: str, glb_path: str):
    textures = {
        'albedo': obj_path.replace('.obj', '.jpg'),# 漫反射/基础颜色
        'metallic': obj_path.replace('.obj', '_metallic.jpg'),# metallic（金属度贴图）
        'roughness': obj_path.replace('.obj', '_roughness.jpg')# roughness（粗糙度贴图）
        }
    create_glb_with_pbr_materials(obj_path, textures, glb_path)

def gen_save_folder(uid,save_dir='./text23d_save_dir',max_size=200):
    # 检查 save_dir 是否已经包含 uid（避免重复创建子目录）
    if save_dir.endswith(str(uid)) or os.path.basename(save_dir) == str(uid):
        # save_dir 已经是完整路径，直接使用
        os.makedirs(save_dir, exist_ok=True)
        print(f"Using existing folder: {save_dir}")
        return save_dir
    
    # save_dir 是基础目录，需要管理子目录和清理旧目录
    os.makedirs(save_dir, exist_ok=True)
    dirs = [f for f in Path(save_dir).iterdir() if f.is_dir()]
    if len(dirs) >= max_size:
        oldest_dir = min(dirs, key=lambda x: x.stat().st_ctime)
        shutil.rmtree(oldest_dir)
        print(f"Removed the oldest folder: {oldest_dir}")
    new_folder = os.path.join(save_dir, str(uid))
    os.makedirs(new_folder, exist_ok=True)
    print(f"Created new folder: {new_folder}")
    return new_folder

class Text2ImagePipeline:
    """千问文生图Pipeline封装 - 使用本地模型"""
    
    def __init__(self, model_name: str = "Qwen/Qwen-Image", 
                 device: str = None , 
                 low_vram_mode: bool =False ,
                 local_model_path: str = "text_to_3d_agent/models",
                 GEMINI_API_KEY:str = None):
        """
        初始化千问文生图pipeline
        
        Args:
            model_name: 模型名称，默认为 "Qwen/Qwen-Image"
            device: 计算设备 ('cuda' or 'cpu'),如果为None则自动检测
        """
        self.model_name = model_name
        self.low_vram_mode = low_vram_mode
        self.local_model_path = local_model_path
        # 自动检测设备
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                self.torch_dtype = torch.bfloat16
            else:
                self.device = "cpu"
                self.torch_dtype = torch.float32
        else:
            self.device = device
            self.torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self.offload_device = "cpu"
        self.offload_dtype = torch.float8_e4m3fn
        self.GEMINI_API_KEY = GEMINI_API_KEY
        # 初始化pipeline
        self._init_pipeline(self.low_vram_mode)
        
    def _init_pipeline(self,low_vram_mode):
        if self.model_name == "Qwen/Qwen-Image":
            if low_vram_mode:
                self.pipe = QwenImagePipeline.from_pretrained(
                    torch_dtype=self.torch_dtype,
                    device=self.device,
                    model_configs=[
                        ModelConfig(model_id="Qwen/Qwen-Image", 
                                    origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", 
                                    local_model_path=self.local_model_path,
                                    offload_device=self.offload_device, offload_dtype=self.offload_dtype),
                        ModelConfig(model_id="Qwen/Qwen-Image", 
                                    origin_file_pattern="text_encoder/model*.safetensors", 
                                    local_model_path=self.local_model_path,
                                    offload_device=self.offload_device, offload_dtype=self.offload_dtype),
                        ModelConfig(model_id="Qwen/Qwen-Image", 
                                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors", 
                                    local_model_path=self.local_model_path,
                                    offload_device=self.offload_device, offload_dtype=self.offload_dtype),
                    ],
                    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
                )
                self.pipe.enable_vram_management()
            else:
                self.pipe = QwenImagePipeline.from_pretrained(
                    torch_dtype=self.torch_dtype,
                    device=self.device,
                    model_configs=[
                        ModelConfig(model_id="Qwen/Qwen-Image", 
                                    origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                                    local_model_path=self.local_model_path,
                                    ),
                        ModelConfig(model_id="Qwen/Qwen-Image", 
                                    origin_file_pattern="text_encoder/model*.safetensors",
                                    local_model_path=self.local_model_path,
                                    ),
                        ModelConfig(model_id="Qwen/Qwen-Image", 
                                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                                    local_model_path=self.local_model_path,
                                    ),
                    ],
                    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
                )
        elif self.model_name == "nano-banana":
            # if self.GEMINI_API_KEY != None:
            #     os.environ["GEMINI_API_KEY"] = self.GEMINI_API_KEY
            # load_dotenv()
            # api_key = os.getenv("GEMINI_API_KEY")
            # self.pipe = genai.Client(api_key=api_key)
            self.pipe = OpenAI(api_key=self.GEMINI_API_KEY, base_url="https://hiapi.online/v1")

    def __call__(self, prompt:str, image:Union[Image.Image,str,None] = None,**kwargs):
        # 设置默认参数
        width = kwargs.get('width', 1328)
        height = kwargs.get('height', 1328)
        num_inference_steps = kwargs.get('num_inference_steps', 30)
        guidance_scale = kwargs.get('guidance_scale', 5.0) #引导尺度
        seed = kwargs.get('seed', 42)
        negative_prompt = kwargs.get('negative_prompt', "")
        temperature=kwargs.get('temperature', 0.01)  #控制随机性，越小越确定
        top_p=kwargs.get('top_p', 0.98)  #核心采样策略，取概率累计达到 top_p 的 token 时停止
        # 生成图像
        if self.model_name == "Qwen/Qwen-Image":
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                cfg_scale=guidance_scale,
                seed=seed,
            )
        elif self.model_name == "nano-banana":
            # if image == None:
            #     prompt=prompt
            #     result = self.pipe.models.generate_content(
            #         model="gemini-2.5-flash-image-preview",
            #         contents=[prompt],
            #         config=types.GenerateContentConfig(
            #             temperature=temperature,
            #             top_p=top_p,
            #             seed=seed
            #         )
            #     )
            # else:
            #     if isinstance(image,str):
            #         with open(image, 'rb') as img_file:  
            #             img_data = base64.b64encode(img_file.read()).decode()  
            #     elif isinstance(image,Image.Image):
            #         buffer = io.BytesIO()
            #         image.save(buffer, format="PNG")  # 你也可以改成 "JPEG"
            #         img_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
            #     result = self.pipe.models.generate_content(
            #         model="gemini-2.5-flash-image-preview",
            #         contents=[{'inline_data': {'mime_type': 'image/jpeg', 'data': img_data}},
            #                   prompt],
            #         config=types.GenerateContentConfig(
            #             temperature=temperature,
            #             top_p=top_p,
            #             seed=seed
            #         )
            #     )
            # for idx, part in enumerate(result.candidates[0].content.parts):
            #     if getattr(part, "inline_data", None):
            #         result = part.inline_data.data
            #         result = Image.open(io.BytesIO(result))
            messages = []
            if image is None:
                messages.append({"role": "user", "content": prompt})
            else:
                img_data = None
                if isinstance(image, str):
                    with open(image, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                elif isinstance(image, Image.Image):
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    img_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                if img_data:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
                        ]
                    })
                else:
                    messages.append({"role": "user", "content": prompt})

            response = self.pipe.chat.completions.create(
                model="gemini-2.5-flash-image-preview",
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                stream=False
            )
            
            # Assuming the response contains base64 encoded image in the content
            # This part is speculative as the OpenAI API for chat does not natively return images.
            # This assumes the custom base_url provides this functionality.
            result = None
            if response.choices[0].message.content:
                message_content = response.choices[0].message.content
                image_url = None
                
                # Check for markdown image syntax ![...](...) and extract URL
                import re
                match = re.search(r'\((https?://[^\)]+)\)', message_content)
                if match:
                    image_url = match.group(1)
                elif "http" in message_content: # Fallback if just a URL is returned
                    # Extract URL if it's not in markdown format but exists
                    url_match = re.search(r'(https?://[^\s]+)', message_content)
                    if url_match:
                        image_url = url_match.group(0)

                if image_url:
                    try:
                        import requests
                        from io import BytesIO
                        # Some servers might block requests without a user-agent
                        headers = {'User-Agent': 'Mozilla/5.0'}
                        img_response = requests.get(image_url, headers=headers)
                        img_response.raise_for_status()
                        result = Image.open(BytesIO(img_response.content))
                        print(f"Successfully downloaded image from {image_url}")
                    except Exception as url_e:
                        print(f"Could not download image from URL '{image_url}': {url_e}")
                else:
                    # Try decoding as base64 as a last resort, as it was failing before
                    try:
                        image_data = base64.b64decode(message_content)
                        result = Image.open(io.BytesIO(image_data))
                    except Exception as e:
                        print(f"Could not decode image from response (Base64 attempt failed): {e}")

        # 返回生成的图像
        return result
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'pipe'):
            del self.pipe
        if self.device == "cuda":
            torch.cuda.empty_cache()

class Text23dPipeline:
    def __init__(self,
                 low_vram_mode: bool =False, 
                 device='cuda', 
                 hunyuan_file_path: str ="text_to_3d_agent/Hunyuan3D-2.1",
                 hunyuan_model_path: str = "tencent/Hunyuan3D-2.1",
                 hunyuan_subfolder: str = "hunyuan3d-dit-v2-1",
                 qwen_model_path: str ="text_to_3d_agent/models",
                 txt2img_model_name: str = "Qwen/Qwen-Image",
                 GEMINI_API_KEY: str = None):
        """
        初始化Hunyuan3D Agent
        
        Args:
            device: 计算设备 ('cuda' or 'cpu')
            qwen_model_name: 千问模型名称
        """
        self.device = device
        self.low_vram_mode = low_vram_mode
        self.txt2img_model_name = txt2img_model_name
        self.hunyuan_file_path = hunyuan_file_path
        self.hunyuan_model_path = hunyuan_model_path
        self.hunyuan_subfolder = hunyuan_subfolder
        self.qwen_model_path = qwen_model_path
        self.GEMINI_API_KEY = GEMINI_API_KEY 
        self._init_pipelines()
        
    def _init_pipelines(self):
        """初始化所有pipeline"""
        # 设置模型基础目录环境变量（使用绝对路径）
        models_base_dir = str(PROJECT_ROOT / 'text_to_3d_agent' / 'models')
        os.environ['HY3DGEN_MODELS'] = models_base_dir
        print(f"Set HY3DGEN_MODELS to: {models_base_dir}")
        
        # 跳过文生图pipeline初始化，直接使用输入图像
        print("Skipping text2image pipeline initialization - using direct image input")
        self.text2image_pipeline = None

        # 混元3D形状生成pipeline
        self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path= self.hunyuan_model_path,
            subfolder=self.hunyuan_subfolder,
            use_safetensors=False,
            device=self.device,
        )
        print("Successfully initialized Hunyuan3DDiTFlowMatching pipeline")
        
        self.rembg = BackgroundRemover()# 背景移除器
        
        # 配置纹理生成pipeline，优化多视角支持
        conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)  # 增加最大视角数
        # 使用绝对路径
        hunyuan_abs_path = str(PROJECT_ROOT / self.hunyuan_file_path)
        conf.realesrgan_ckpt_path = os.path.join(hunyuan_abs_path, "hy3dpaint","ckpt", "RealESRGAN_x4plus.pth")
        conf.multiview_cfg_path = os.path.join(hunyuan_abs_path, "hy3dpaint","cfgs", "hunyuan-paint-pbr.yaml")
        conf.custom_pipeline = os.path.join(hunyuan_abs_path, "hy3dpaint","hunyuanpaintpbr")
        
        # 纹理生成pipeline
        self.paint_pipeline = Hunyuan3DPaintPipeline(conf)
        print("Successfully initialized Hunyuan3DPaintPipeline pipeline")
    
    def generate_2d(self,
                    uid: int,
                    input_source: str,  # 可以是图像路径或文本描述
                    save_dir: str ='text_to_3d_agent/text23d_save_dir',
                   **kwargs):
        if input_source: print('prompt is:', input_source)
        self.save_dir = gen_save_folder(uid,save_dir)
        input_dir = os.path.join(self.save_dir, "input")

        os.makedirs(input_dir, exist_ok=True)
        
        # 直接使用输入的图像路径
        images_path = os.path.join(input_dir, 'input.png')
        
        if input_image and os.path.exists(input_image):
            # 使用提供的图像路径
            print(f"Using provided input image for 2D generation: {input_image}")
            images = Image.open(input_image)
            # 保存到input目录
            images.save(images_path)
        else:
            raise ValueError(f"Input image is required for 2D generation but not provided or not found: {input_image}")
            
        return images, images_path   
     
    def generate_3d(self, 
                   uid: int,
                   input_source: str,  # 可以是图像路径或文本描述
                   input_image:Union[Image.Image,str,None] = None,
                   save_dir: str ='text_to_3d_agent/text23d_save_dir',
                   **kwargs) -> Dict[str, Any]:
        
        # images,images_path = self.generate_2d(uid,input_source,save_dir,**kwargs)
        if input_source: print('prompt is:', input_source)
        self.save_dir = gen_save_folder(uid,save_dir)
        input_dir = os.path.join(self.save_dir, "input")

        os.makedirs(input_dir, exist_ok=True)
        # 直接使用输入的图像路径
        if input_image and os.path.exists(input_image):
            # 使用提供的图像路径
            print(f"Using provided input image: {input_image}")
            images = Image.open(input_image)
            # 保存到input目录
            images.save(os.path.join(input_dir, 'input.png'))
        else:
            raise ValueError(f"Input image is required but not provided or not found: {input_image}")

        ############################ Image  To   3D ##################################

        image23d_params = kwargs.get('text2image_params', {})
        steps = image23d_params.get('steps', 50)
        guidance_scale = image23d_params.get('guidance_scale', 7.5)
        seed = image23d_params.get('seed', 42)
        octree_resolution = image23d_params.get('octree_resolution', 256)
        check_box_rembg = image23d_params.get('check_box_rembg', False)
        num_chunks = image23d_params.get('num_chunks ', 200000)

        # remove the background
        if check_box_rembg or images.mode == "RGB":
            images = self.rembg(images)
            images.save(os.path.join(input_dir, 'rembg.png'))

        # image to white model
        generator = torch.Generator()
        generator = generator.manual_seed(int(seed))
        outputs = self.shape_pipeline(
            image=images,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            output_type='mesh'
        )
        mesh = export_to_trimesh(outputs)[0]

        # Export initial mesh without texture
        shape_dir = os.path.join(self.save_dir, "3Dshape")
        os.makedirs(shape_dir, exist_ok=True)  # 确保目录存在

        initial_save_path = os.path.join(shape_dir, f'{str(uid)}_initial.glb')
        mesh.export(initial_save_path)

        # Generate textured mesh as obj ( as in demo )
        main_image = images 
        paint_dir = os.path.join(self.save_dir, "3Dpaint")
        os.makedirs(paint_dir, exist_ok=True)  # 确保目录存在
        
        output_mesh_path_obj = os.path.join(paint_dir, f'{str(uid)}_texturing.obj')
        textured_path_obj = self.paint_pipeline(
            mesh_path=initial_save_path,
            image_path=main_image,
            output_mesh_path=output_mesh_path_obj,
            save_glb=False            
        )

        # Use the textured GLB as the final output
        #final_save_path = os.path.join(self.save_dir, f'{str(uid)}_textured.{file_type}')
        #os.rename(output_mesh_path, final_save_path)

        # Convert textured OBJ to GLB using obj2gltf with PBR support
        print("convert textured OBJ to GLB")
        glb_path_textured = os.path.join(paint_dir, f'{str(uid)}_texturing.glb')
        quick_convert_with_obj2gltf(textured_path_obj, glb_path_textured)

        # now rename glb_path to uid_textured.glb
        print("done.")
        output_dir = os.path.join(self.save_dir, "output")
        os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

        final_save_path = os.path.join(output_dir, f'{str(uid)}_textured.glb')
        os.rename(glb_path_textured, final_save_path)
        print(f"final_save_path: {final_save_path}")

        mr_path = "mr_combined.png"
        if os.path.exists(mr_path):
            os.remove(mr_path)
        temp_path =  "temp.glb"
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return images, final_save_path

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'low_vram_mode') and self.low_vram_mode:
            torch.cuda.empty_cache()
            gc.collect()                # 强制垃圾回收