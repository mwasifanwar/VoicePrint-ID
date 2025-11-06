# main.py
import argparse
import yaml
from voiceprint_id.utils.config_loader import ConfigLoader

def main():
    parser = argparse.ArgumentParser(description='VoicePrint ID: Multi-Speaker Recognition System')
    parser.add_argument('--mode', type=str, choices=['api', 'train', 'inference', 'dashboard'], required=True)
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--audio', type=str, help='Input audio path for inference')
    parser.add_argument('--analysis', type=str, choices=['all', 'speaker', 'emotion', 'language', 'spoof'], help='Analysis type')
    
    args = parser.parse_args()
    
    config_loader = ConfigLoader(args.config)
    
    if args.mode == 'api':
        print("Starting VoicePrint ID API Server...")
        from voiceprint_id.api.fastapi_server import FastAPIServer
        server = FastAPIServer(config_loader)
        server.run()
    
    elif args.mode == 'train':
        print("Starting model training...")
        from train import main as train_main
        train_main()
    
    elif args.mode == 'inference':
        print("Running inference...")
        from inference import main as inference_main
        
        if not args.audio:
            print("Error: --audio is required for inference mode")
            return
        
        inference_args = argparse.Namespace()
        inference_args.audio = args.audio
        inference_args.analysis = args.analysis or 'all'
        inference_args.output = f"voiceprint_results_{args.analysis or 'all'}.json"
        
        inference_main()
    
    elif args.mode == 'dashboard':
        print("Starting VoicePrint ID Dashboard...")
        from voiceprint_id.dashboard.app import create_app
        app = create_app()
        app.run(
            host=config_loader.get('dashboard.host', '0.0.0.0'),
            port=config_loader.get('dashboard.port', 5000),
            debug=config_loader.get('dashboard.debug', True)
        )

if __name__ == "__main__":
    main()