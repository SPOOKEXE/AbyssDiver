
from PIL import Image
from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional, Union
from urllib.parse import urlencode
from uuid import uuid4

import aiohttp
import asyncio
import base64
import json
import traceback
import uvicorn
import websockets

### COMFYUI.py ###
COMFYUI_IMAGE_TYPE = Literal["input", "output", "temp"]

COMFYUI_SAMPLERS = Literal[
	"euler", "euler_ancestral", "heun",
	"dpm_2", "dpm_2_ancestral", "lms",
	"ddpm", "ddim", "uni_pc"
]

COMFYUI_SCHEDULERS= Literal[
	"normal", "karras", "exponential",
	"simple", "ddim_uniform", "beta"
]

def image_to_base64(image : Image.Image) -> str:
	buffered = BytesIO()
	image.save(buffered, format="PNG")
	return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_image(b64 : str) -> Image.Image:
	buffer = BytesIO(base64.b64decode(b64))
	return Image.open(buffer).convert('RGB')

async def async_post(url : str, headers : Optional[dict] = None, cookies : Optional[dict] = None, json : Optional[dict] = None, data : Optional[str] = None) -> bytes:
	'''Asynchronously POST to the given url with the parameters.'''
	client : aiohttp.ClientSession
	async with aiohttp.ClientSession(headers=headers, cookies=cookies) as client:
		response : aiohttp.ClientResponse = await client.post(url, data=data, json=json)
		return await response.read()

async def post_json_response(url : str, data : Optional[dict]) -> Union[dict, list, None]:
	try:
		response : bytes = await async_post(url, json=data)
		return json.loads(response.decode('utf-8'))
	except Exception as e:
		traceback.print_exception(e)
		return None

async def async_get(url : str, headers : Optional[dict] = None, cookies : Optional[dict] = None, json : Optional[dict] = None, data : Optional[str] = None) -> bytes:
	'''Asynchronously POST to the given url with the parameters.'''
	client : aiohttp.ClientSession
	async with aiohttp.ClientSession(headers=headers, cookies=cookies) as client:
		response : aiohttp.ClientResponse = await client.get(url, data=data, json=json)
		return await response.read()

async def get_json_response(url : str) -> Union[dict, list, None]:
	try:
		response : bytes = await async_get(url)
		return json.loads(response.decode('utf-8'))
	except Exception as e:
		traceback.print_exception(e)
		return None

class ComfyUI_API:
	'''
	server_address : str = "127.0.0.1:8188"

	Cleaned up and asynchronous version of:
	- https://github.com/9elements/comfyui-api/blob/main/basic_api.py
	'''
	server_address : str
	client_id : str

	_active_ids : dict[str, bool]
	_websocket : Optional[websockets.WebSocketClientProtocol]

	def __init__(self, server_address : str) -> None:
		self.server_address = server_address
		self.client_id = uuid4().hex
		self._active_ids = dict()

		print(self.client_id)

	async def open_websocket(self) -> None:
		address : str = f"ws://{self.server_address}/ws?clientId={self.client_id}"
		self._websocket = websockets.connect(address)

	async def close_websocket(self) -> None:
		self._websocket = None

	async def queue_prompt(self, prompt : dict) -> str:
		'''Queue the given prompt and return a prompt_id'''
		payload = {"prompt": prompt, "client_id": self.client_id}
		response = await post_json_response(f"http://{self.server_address}/prompt", data=payload)
		prompt_id : str = response['prompt_id']
		print(prompt_id)
		self._active_ids[prompt_id] = False
		return prompt_id

	async def is_prompt_id_finished(self, prompt_id : str) -> Optional[bool]:
		'''Check if the given prompt_id is finished.'''
		return self._active_ids.get(prompt_id)

	async def await_prompt_id(self, prompt_id : str) -> Optional[bool]:
		'''Await for the prompt id to finish - also returns if it finished or not.'''
		finished : Optional[bool] = self.is_prompt_id_finished(prompt_id)
		while finished is False:
			await asyncio.sleep(1.0)
			finished = self.is_prompt_id_finished(prompt_id)
		return finished

	async def fetch_prompt_id_history(self, prompt_id : str) -> dict:
		'''Fetch the generation history for the given prompt_id.'''
		history = await get_json_response(f"http://{self.server_address}/history/{prompt_id}")
		return history.get(prompt_id)

	async def fetch_image(self, filename : str, subfolder : str, folder_type : str) -> bytes:
		payload = {"filename": filename, "subfolder": subfolder, "type": folder_type}
		query : str = urlencode(payload)
		response : bytes = await async_get(f"http://{self.server_address}/view?{query}")
		return response

	async def fetch_prompt_id_images(self, prompt_id : str, include_previews : bool = False) -> list[dict]:
		'''Fetch the generated images for the given prompt_id if any.'''
		images : list[dict] = list()
		history : dict = await self.fetch_prompt_id_history(prompt_id)
		for node_id in history['outputs']:
			node_output = history['outputs'][node_id]
			if 'images' not in node_output:
				continue
			for image in node_output['images']:
				output_data = {"node_id" : node_id, "file_name" : image["filename"], "type" : image["type"]}
				if include_previews is True and image['type'] == 'temp':
					preview_data : bytes = await self.fetch_image(image['filename'], image['subfolder'], image['type'])
					output_data['image_data'] = preview_data
				if image['type'] == 'output':
					image_data : bytes = await self.fetch_image(image['filename'], image['subfolder'], image['type'])
					output_data['image_data'] = image_data
				images.append(output_data)

		return images

	async def track_progress(self, prompt_id : str, node_ids : list[int]) -> None:
		'''Echo the progress of the prompt_id.'''
		finished_nodes : list[str] = []
		async with self._websocket as socket:
			while True:
				# receive content
				content = await socket.recv()
				if isinstance(content, str) is False:
					continue
				message = json.loads(content)
				# progression of current
				if message['type'] == 'progress':
					data = message['data']
					current_step = data['value']
					print('In K-Sampler -> Step: ', current_step, ' of: ', data['max'])
				# another step of execution done
				if message['type'] == 'execution_cached':
					data = message['data']
					for itm in data['nodes']:
						if itm not in finished_nodes:
							finished_nodes.append(itm)
							print('Progess: ', len(finished_nodes)-1, '/', len(node_ids), ' Tasks done')
				# executing a new node if any
				if message['type'] == 'executing':
					data = message['data']
					if data['node'] not in finished_nodes:
						finished_nodes.append(data['node'])
						print('Progess: ', len(finished_nodes)-1, '/', len(node_ids), ' Tasks done')
					if data['node'] is None and data['prompt_id'] == prompt_id:
						# execution is done
						self._active_ids[prompt_id] = True
						break

	async def cleanup_prompt_id(self, prompt_id : str) -> None:
		self._active_ids.pop(prompt_id, None)

	async def generate_images_using_worflow_prompt(self, prompt : dict, include_previews : bool = True) -> list[dict]:
		'''Complete the full sequence of giving a prompt and receiving the images.'''
		prompt_id : str = await self.queue_prompt(prompt)
		await self.track_progress( prompt_id, prompt.keys() )
		images : list[dict] = await self.fetch_prompt_id_images(prompt_id, include_previews=include_previews)
		await self.cleanup_prompt_id(prompt_id)
		return images

	async def upload_image(self, image : Image.Image, save_name : str, image_type : COMFYUI_IMAGE_TYPE = "input", overwrite : bool = True) -> bool:
		"""Upload an image to ComfyUI to be used for workflows."""
		# use 'save_name' in LoadImage objects (with extension)
		# prepare image data
		byte_io = BytesIO()
		image.save(byte_io, format='PNG')
		byte_data = byte_io.getvalue()
		# prepare form data
		data = aiohttp.FormData()
		data.add_field('image', byte_data, filename=save_name, content_type="image/png")
		data.add_field('type', image_type)
		data.add_field('overwrite', str(overwrite).lower())
		# send request
		try:
			url : str = f'http://{self.server_address}/upload/image'
			client : aiohttp.ClientSession
			async with aiohttp.ClientSession() as client:
				response : aiohttp.ClientResponse = await client.post(url, data=data)
				if response.status != 200:
					print(f"Failed to upload image due to: {response.reason} (typicallycfile data)")
			return True
		except Exception as e:
			traceback.print_exception(e)
			return False
##################

### MODELS.py ###
class CharacterInfo(BaseModel):
	id : int
	name : str
	cost : int
	carry : int
	affec : int
	swap : bool
	mindSex : str
	osex : str
	obreasts : int
	desiredBreasts : int
	openis : int
	ogender : int
	fit : int
	oheight : int
	comfortableHeight : int
	age : int
	appDesc : str
	fear : str
	ohair : str
	oskinColor : str
	oskinType : str
	oears : str
	oeyeColor : str
	oblood : str
	pregnantT : int
	due : int
	tentaclePreg : bool
	lastBirth : int
	switched : bool
	gestationJumps : int
	location : int

class CharacterState(BaseModel):
	apparent_age : int | float
	apparent_gender : int | float
	armCount : int
	blood : str
	bodyHair : int
	breasts : int | float
	breastsLabel : str
	description : str
	double_penis : int
	ears : str
	extraEyes : int
	extraMouths : int
	eyeColor : str
	eyeCount : int
	fluids : int
	genderVoice : int
	hair : str
	height : int | float
	horns : int
	inhuman : int
	lactation : int
	legCount : int
	lewdness : int
	libido : int
	penis_size : int
	real_age : int | float
	real_gender : int | float
	sex : str
	skinColor : str
	skinType : str
	subdom : int
	tail : list[str]
	tentacles : int
	vagina_count : int
	wombs : int

class CharacterData(BaseModel):
	character : CharacterInfo
	curses : list[str]
	state : CharacterState

class SceneData(BaseModel):
	scene_id : str
	scene_params : list
###############################

### WORKFLOWS ###
SDXL_MODELS = Literal["hassakuXLHentai_v13.safetensors"]
SDXL_DEPTH_CONTROL_MODELS = Literal["control-lora-depth-rank256.safetensors"]

SIMPLE_TXT2IMG_IMAGE_GENERIC_WORKFLOW : dict = {
	"5": {
		"inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
		"class_type": "CheckpointLoaderSimple",
		"_meta": {"title": "Load Checkpoint"}
	},
	"9": {
		"inputs": {"text": "","clip": ["5",1]},
		"class_type": "CLIPTextEncode",
		"_meta": {"title": "Positive Prompt"}
	},
	"10": {
		"inputs": {"text": "","clip": ["5",1]},
		"class_type": "CLIPTextEncode",
		"_meta": {"title": "Negative Prompt"}
	},
	"11": {
		"inputs": {
			"seed": 961176784184834,
			"steps": 35,
			"cfg": 7,
			"sampler_name": "dpmpp_3m_sde_gpu",
			"scheduler": "exponential",
			"denoise": 1,
			"model": ["5",0],
			"positive": ["9",0],
			"negative": ["10",0],
			"latent_image": ["12",0]
		},
		"class_type": "KSampler",
		"_meta": {"title": "KSampler"}
	},
	"12": {
		"inputs": {"width": 1024,"height": 1024,"batch_size": 1},
		"class_type": "EmptyLatentImage",
		"_meta": {"title": "Empty Latent Image"}
	},
	"15": {
		"inputs": {"samples": ["11",0],"vae": ["5",2]},
		"class_type": "VAEDecode",
		"_meta": {"title": "VAE Decode"}
	},
	"16": {
		"inputs": {"filename_prefix": "ComfyUI","images": ["15",0]},
		"class_type": "SaveImage",
		"_meta": {"title": "Save Image"}
	}
}

CONTROL_NET_DYNAMIC_DEPTH_WORKFLOW : dict = {
	"1": {
		"inputs": {"image": "00007-4032934194.png","upload": "image"},
		"class_type": "LoadImage",
		"_meta": {"title": "Load Image"}
	},
	"2": {
		"inputs": {"a": 6.2832,"bg_threshold": 0.1,"resolution": 512,"image": ["1",0]},
		"class_type": "MiDaS-DepthMapPreprocessor",
		"_meta": {"title": "MiDaS Depth Map"}
	},
	"3": {
		"inputs": {"images": ["2",0]},
		"class_type": "PreviewImage",
		"_meta": {"title": "Preview Image"}
	},
	"5": {
		"inputs": {"ckpt_name": "ThinkDiffusionXL.safetensors"},
		"class_type": "CheckpointLoaderSimple",
		"_meta": {"title": "Load Checkpoint"}
	},
	"7": {
		"inputs": {"control_net_name": "control-lora-depth-rank256.safetensors"},
		"class_type": "ControlNetLoader",
		"_meta": {"title": "Load ControlNet Model"}
	},
	"8": {
		"inputs": {
			"strength": 1,
			"start_percent": 0,
			"end_percent": 1,
			"positive": ["9",0],
			"negative": ["10",0],
			"control_net": ["7",0],
			"image": ["2",0]
		},
		"class_type": "ControlNetApplyAdvanced",
		"_meta": {"title": "Apply ControlNet (Advanced)"}
	},
	"9": {
		"inputs": {"text": "a futuristic cyborg on an alien spaceship","clip": ["5",1]},
		"class_type": "CLIPTextEncode",
		"_meta": {"title": "Positive Prompt"}
	},
	"10": {
		"inputs": {"text": "","clip": ["5",1]},
		"class_type": "CLIPTextEncode",
		"_meta": {"title": "Negative Prompt"}
	},
	"11": {
		"inputs": {
			"seed": 961176784184834,
			"steps": 35,
			"cfg": 7,
			"sampler_name": "dpmpp_3m_sde_gpu",
			"scheduler": "exponential",
			"denoise": 1,
			"model": ["5",0],
			"positive": ["8",0],
			"negative": ["8",1],
			"latent_image": ["12",0]
		},
		"class_type": "KSampler",
		"_meta": {"title": "KSampler"}
	},
	"12": {
		"inputs": {"width": 1024,"height": 1024,"batch_size": 1},
		"class_type": "EmptyLatentImage",
		"_meta": {"title": "Empty Latent Image"}
	},
	"15": {
		"inputs": {"samples": ["11",0], "vae": ["5",2]},
		"class_type": "VAEDecode",
		"_meta": {"title": "VAE Decode"}
	},
	"16": {
		"inputs": {"filename_prefix": "ComfyUI", "images": ["15",0]},
		"class_type": "SaveImage",
		"_meta": {"title": "Save Image"}
	}
}

# SIMPLE_TXT2IMG_IMAGE_GENERIC_WORKFLOW
class PortraitT2IGenericWorkflow(BaseModel):
	checkpoint : SDXL_MODELS
	positive_prompt : str
	negative_prompt : str
	steps : int = 20
	cfg : Union[float, int] = 7.0
	width : int = 1024
	height : int = 1024
	seed : int = 0

# CONTROL_NET_DYNAMIC_DEPTH_WORKFLOW
class PortraitT2IDepthControlWorkflow(BaseModel):
	checkpoint : SDXL_MODELS
	positive_prompt : str
	negative_prompt : str
	controlnet_depth_model : SDXL_DEPTH_CONTROL_MODELS
	MiDaS_depth_preprocess_image_base64 : str
	steps : int = 25
	cfg : Union[int, float] = 7.0
	width : int = 1024
	height : int = 1024
	seed : int = -1

async def PrepareSimpleT2IWorkflow(params : PortraitT2IGenericWorkflow) -> dict:
	workflow = SIMPLE_TXT2IMG_IMAGE_GENERIC_WORKFLOW.copy()
	workflow["5"]["inputs"]["ckpt_name"] = params.checkpoint
	workflow["9"]["inputs"]["text"] = params.positive_prompt
	workflow["10"]["inputs"]["text"] = params.negative_prompt
	workflow["11"]["inputs"]["steps"] = params.steps
	workflow["11"]["inputs"]["cfg"] = params.cfg
	workflow["11"]["inputs"]["seed"] = params.seed
	workflow["12"]["inputs"]["width"] = params.width
	workflow["12"]["inputs"]["height"] = params.height
	return workflow

async def PrepareSDXLDepthWorkflow(params : PortraitT2IDepthControlWorkflow) -> dict:
	workflow = CONTROL_NET_DYNAMIC_DEPTH_WORKFLOW.copy()
	workflow["5"]["inputs"]["ckpt_name"] = params.checkpoint
	workflow["7"]["inputs"]["control_net_name"] = params.controlnet_depth_model
	workflow["9"]["inputs"]["text"] = params.positive_prompt
	workflow["10"]["inputs"]["text"] = params.negative_prompt
	workflow["11"]["inputs"]["seed"] = params.seed
	workflow["11"]["inputs"]["steps"] = params.steps
	workflow["11"]["inputs"]["cfg"] = params.cfg
	workflow["12"]["inputs"]["width"] = params.width
	workflow["12"]["inputs"]["height"] = params.height
	return workflow
####################

### game.py ###
DEFAULT_POSITIVE_PROMPT : str = "(masterpiece,best quality,high quality,medium quality,normal quality)"
DEFAULT_NEGATIVE_PROMPT : str = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

BODY_FITNESS : list[str] = ["fragile body", "weak body", "average body", "fit body", "very fit body"]
HEIGHT_RANGES : list[tuple[str, int]] = [("dwarf",150),("midget",160),("short",170),("",183),("tall",195)]

# GENDER_REVERSAL_STAGE = ["", "dainty, twink,", "androgynous", "feminine", "female"]
PENIS_SIZES : list[str] = ["small penis", "below average penis", "average penis", "large penis", "huge penis"]

def height_to_ranged_value(height : int | float) -> str:
	for item in HEIGHT_RANGES:
		if height < item[1]:
			return item[0]
	return HEIGHT_RANGES[len(HEIGHT_RANGES)-1][0]

# PORTRAIT GENERATOR
async def prepare_portrait_prompt(character_data : CharacterData) -> str:
	base_prompt = DEFAULT_POSITIVE_PROMPT
	base_prompt += ",solo,portrait,upper_body,plain dark background,"
	# temporary
	base_prompt += character_data.state.sex + ","
	base_prompt += f"{max(character_data.state.apparent_age, 21)} years old,"
	base_prompt += BODY_FITNESS[max(min(character_data.character.fit+2, 4), 0)] + ","
	base_prompt += height_to_ranged_value(character_data.state.height) + ","
	base_prompt += character_data.state.hair + " hair,"
	for tail in character_data.state.tail:
		base_prompt += character_data.state.hair + " " + tail + " tail,"
	base_prompt += character_data.state.hair + " " + character_data.state.ears + " ears,"
	# CreatureOfTheNight -> Vampire
	if "CreatureOfTheNight" in character_data.curses:
		base_prompt += "vampire,fangs,red eyes,glowing eyes,pale skin,"
	else:
		base_prompt += character_data.state.eyeColor + " eyes,"
		base_prompt += character_data.state.skinColor + " skin,"
	if "WrigglyAntennae" in character_data.curses:
		base_prompt += "pink antennae,"
	if "Megadontia" in character_data.curses:
		base_prompt += "sharp teeth,"
	if "FreckleSpeckle" in character_data.curses:
		base_prompt += "freckles,"
	if "KnifeEar" in character_data.curses:
		base_prompt += "pointy ears,"
	if "Horny" in character_data.curses:
		base_prompt += "succubus horns,"
	if "DrawingSpades" in character_data.curses:
		base_prompt += "spade tail,"
	if "ClothingRestrictionA" not in character_data.curses:
		base_prompt += "earrings,"
	if "ClothingRestrictionC" not in character_data.curses:
		base_prompt += "adventurer,leather armor,"
	if "ClothingRestrictionC" in character_data.curses:
		if "ClothingRestrictionB" in character_data.curses:
			base_prompt += "nude,"
		else:
			if character_data.state.sex == "female":
				base_prompt += "bra,panties,"
			else:
				base_prompt += "underwear,shirtless,no pants,"
				base_prompt += "small penis bulge,"
	if "ClothingRestrictionC" in character_data.curses and "ClothingRestrictionB" in character_data.curses:
		# NUDE
		if "Null" in character_data.curses:
			# null curse
			base_prompt += "smooth featureless body, no genitalia, soft abstract body aesthetic without explicit details,"
		else:
			# sex-specific
			if character_data.state.sex == "female":
				# female
				if "TattooTally" in character_data.curses:
					base_prompt += "succubus tattoo,"
				if "Leaky" in character_data.curses:
					base_prompt += "pussy juice,"
			else:
				# male
				if "TattooTally" in character_data.curses:
					base_prompt += "incubus tattoo,"
				if "Leaky" in character_data.curses:
					base_prompt += "pre-ejaculation,"
			# lactation (both M/F)
			lactation : int = 0
			if "LactationRejuvenationA" in character_data.curses:
				lactation += 1
			if "LactationRejuvenationB" in character_data.curses:
				lactation += 1
			if lactation == 2:
				base_prompt += "milk,lactating,lactation,"
			elif lactation == 1:
				base_prompt += "dripping lactation,"
			if character_data.state.penis_size > 0:
				base_prompt += PENIS_SIZES[character_data.state.penis_size-1] + ","
	base_prompt += character_data.state.breastsLabel + " breasts,"
	return base_prompt

async def generate_character(
	character : CharacterData
) -> Optional[Image.Image]:
	prompt : str = await prepare_portrait_prompt(character)
	print(prompt)

	params = PortraitT2IGenericWorkflow(
		checkpoint="hassakuXLHentai_v13.safetensors",
		positive_prompt=prompt,
		negative_prompt=DEFAULT_NEGATIVE_PROMPT
	)
	print(params)

	workflow = await PrepareSimpleT2IWorkflow(params)
	print(workflow)

	COMFYUI_NODE = ComfyUI_API('127.0.0.1:8188')
	await COMFYUI_NODE.open_websocket()

	image_array : list[dict] = await COMFYUI_NODE.generate_images_using_worflow_prompt(workflow)

	await COMFYUI_NODE.close_websocket()

	if len(image_array) == 0: return None

	raw_image : bytes = image_array[0]['image_data']
	return Image.open(BytesIO(raw_image))
####################

app = FastAPI(title="Abyss Diver ComfyUI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/generate_portrait")
async def API_generate_character(
	request : Request,
	character : CharacterData = Body(None, embed=True)
) -> dict:
	print(await request.json())
	try:
		image = await generate_character(character)
	except Exception as e:
		print(e)
		return {'error' : 'Error occured.'}
	return {'images' : [image_to_base64(image)]}

async def main() -> None:
	config = uvicorn.Config(app=app, host='127.0.0.1', port=8000)
	server = uvicorn.Server(config)
	await server.serve()

if __name__ == '__main__':
	asyncio.run(main())
