from converter import Converter
conv = Converter()

info = conv.probe('Rod_vibrarion_mit.cine')

convert = conv.convert('Rod_vibrarion_mit.cine', 'Rod_vibrarion_mit.mp4', {
    'format': 'mp4',
    'audio': {
        'codec': 'aac',
        'samplerate': 11025,
        'channels': 2
    },
    'video': {
        'codec': 'hevc',
        'width': 720,
        'height': 400,
        'fps': 30
    }})

for timecode in convert:
    print(f'\rConverting ({timecode:.2f}) ...')