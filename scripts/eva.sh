  433  torchrun --standalone --nproc_per_node=1 evaluate.py activations --data="result/gt256" --dest=eval-refs/activations_ref.npz --batch=20
  434  torchrun --standalone --nproc_per_node=1 evaluate.py activations --data="result/de256" --dest=eval-refs/activations_sample.npz --batch=20
  #435  python evaluate.py calc -m fid -m pr --activations_sample=eval-refs/activations_sample.npz --activations_ref=eval-refs/activations_ref.npz 
  436  python evaluate.py calc -m fid --activations_sample=eval-refs/activations_sample.npz --activations_ref=eval-refs/activations_ref.npz 