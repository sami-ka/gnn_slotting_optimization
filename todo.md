- once we know what is the real improvement (look at claude's plan), need to have some diversity in what we add to the dataset
- think about ways to generate not completely random orderbook
- look at what part of sample generating takes time
- iterate



Last Claude plan

Implement the following plan:                                                                                            
                                                                                                                           
  # Plan: Add Simulator Verification to Gradient Optimizer                                                                 
                                                                                                                           
  ## Goal                                                                                                                  
  Add verification that the optimized assignment from the GNN actually improves real travel distance as computed by        
  the `Simulator`, not just the GNN's predicted distance.                                                                  
                                                                                                                           
  ## Current State                                                                                                         
  - Gradient optimization in `optimize_slotting.py` reports GNN-predicted improvement                                      
  - The ground truth simulator distance is shown but only for the *original* assignment                                    
  - No comparison of real simulator distance before/after optimization                                                     
                                                                                                                           
  ## Problem                                                                                                               
  The test dataset (`test_dataset_cpu.pt`) only contains graph `Data` objects, not the raw `(OrderBook, ItemLocations,     
  Warehouse)` tuples needed to run the simulator.                                                                          
                                                                                                                           
  ## Solution                                                                                                              
  1. **Modify `train_cpu.py`** to save raw samples alongside graph data                                                    
  2. **Modify `optimize_slotting.py`** to load raw samples and verify with simulator                                       
                                                                                                                           
  ## Implementation Steps                                                                                                  
                                                                                                                           
  ### Step 1: Update `train_cpu.py` to save raw samples                                                                    
  Save a separate file with the raw samples that can be used for simulation verification.                                  
                                                                                                                           
  **File**: `notebooks/train_cpu.py`                                                                                       
                                                                                                                           
  After line 346 (`torch.save(test_dataset, "test_dataset_cpu.pt")`), add:                                                 
  ```python                                                                                                                
  # Save raw samples for simulator verification                                                                            
  test_samples = samples[split_idx:]  # Same indices as test_dataset                                                       
  torch.save(test_samples, "test_samples_cpu.pt")                                                                          
  print("  Saved: test_samples_cpu.pt")                                                                                    
  ```                                                                                                                      
                                                                                                                           
  ### Step 2: Add simulator verification function to `optimize_slotting.py`                                                
  Add a function that takes an assignment and runs the simulator.                                                          
                                                                                                                           
  **File**: `notebooks/optimize_slotting.py`                                                                               
                                                                                                                           
  Add imports at top:                                                                                                      
  ```python                                                                                                                
  import sys                                                                                                               
  import os                                                                                                                
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))                                          
  from slotting_optimization.simulator import Simulator                                                                    
  from slotting_optimization.item_locations import ItemLocations                                                           
  ```                                                                                                                      
                                                                                                                           
  Add verification function:                                                                                               
  ```python                                                                                                                
  def verify_with_simulator(                                                                                               
  original_assignment: np.ndarray,                                                                                         
  optimized_assignment: np.ndarray,                                                                                        
  raw_sample: tuple,  # (OrderBook, ItemLocations, Warehouse)                                                              
  ) -> dict:                                                                                                               
  """Run both assignments through real simulator and compare."""                                                           
  order_book, original_il, warehouse = raw_sample                                                                          
  simulator = Simulator()                                                                                                  
                                                                                                                           
  # Get item IDs and location IDs from original ItemLocations                                                              
  items = original_il.items                                                                                                
  locs_list = [original_il.get_location(item) for item in items]                                                           
                                                                                                                           
  # Original distance (should match ground truth)                                                                          
  orig_dist, _ = simulator.simulate(order_book, warehouse, original_il)                                                    
                                                                                                                           
  # Build new ItemLocations with optimized assignment                                                                      
  # Map assignment indices back to location IDs                                                                            
  storage_locs = [loc for loc in warehouse.locations                                                                       
  if loc not in (warehouse.start_point_id, warehouse.end_point_id)]                                                        
                                                                                                                           
  new_records = []                                                                                                         
  for i, item in enumerate(items):                                                                                         
  new_loc_idx = optimized_assignment[i]                                                                                    
  new_records.append({"item_id": item, "location_id": storage_locs[new_loc_idx]})                                          
                                                                                                                           
  new_il = ItemLocations.from_records(new_records)                                                                         
  opt_dist, _ = simulator.simulate(order_book, warehouse, new_il)                                                          
                                                                                                                           
  improvement = (orig_dist - opt_dist) / orig_dist * 100                                                                   
                                                                                                                           
  return {                                                                                                                 
  "original_sim_distance": orig_dist,                                                                                      
  "optimized_sim_distance": opt_dist,                                                                                      
  "sim_improvement_pct": improvement,                                                                                      
  }                                                                                                                        
  ```                                                                                                                      
                                                                                                                           
  ### Step 3: Update `main()` to call verification                                                                         
  After optimization, load raw samples and verify.                                                                         
                                                                                                                           
  **File**: `notebooks/optimize_slotting.py` - in `main()` function                                                        
                                                                                                                           
  After the optimization result is obtained (around line 669), add:                                                        
  ```python                                                                                                                
  # Verify with real simulator                                                                                             
  try:                                                                                                                     
  raw_samples = torch.load("test_samples_cpu.pt", map_location=device, weights_only=False)                                 
  raw_sample = raw_samples[args.sample_idx]                                                                                
                                                                                                                           
  sim_result = verify_with_simulator(                                                                                      
  result["original_assignment"],                                                                                           
  result["optimized_assignment"],                                                                                          
  raw_sample,                                                                                                              
  )                                                                                                                        
                                                                                                                           
  print("\n" + "=" * 60)                                                                                                   
  print("Simulator Verification:")                                                                                         
  print("=" * 60)                                                                                                          
  print(f"  Original (simulator):  {sim_result['original_sim_distance']:.1f}")                                             
  print(f"  Optimized (simulator): {sim_result['optimized_sim_distance']:.1f}")                                            
  print(f"  Real improvement:      {sim_result['sim_improvement_pct']:.2f}%")                                              
  except FileNotFoundError:                                                                                                
  print("\nNote: test_samples_cpu.pt not found. Re-run train_cpu.py to enable simulator verification.")                    
  ```                                                                                                                      
                                                                                                                           
  ## Files to Modify                                                                                                       
                                                                                                                           
  | File | Change |                                                                                                        
  |------|--------|                                                                                                        
  | `notebooks/train_cpu.py:346-347` | Save raw test samples to `test_samples_cpu.pt` |                                    
  | `notebooks/optimize_slotting.py:11-17` | Add imports for simulator and ItemLocations |                                 
  | `notebooks/optimize_slotting.py:~455` | Add `verify_with_simulator()` function |                                       
  | `notebooks/optimize_slotting.py:~670` | Call verification in `main()` |                                                
                                                                                                                           
  ## Verification                                                                                                          
                                                                                                                           
  1. Re-run training: `cd notebooks && uv run train_cpu.py`                                                                
  - Verify `test_samples_cpu.pt` is created                                                                                
  2. Run optimizer: `uv run optimize_slotting.py --method gradient --sample_idx 0`                                         
  3. Check output shows both:                                                                                              
  - GNN predicted improvement (as before)                                                                                  
  - Real simulator improvement (new)                                                                                       
  4. Compare the two improvements - they should be correlated but may differ                                               
                                                                                                                           
                                                                                                                           
  If you need specific details from before exiting plan mode (like exact code snippets, error messages, or content you     
  generated), read the full transcript at: C:\Users\samik\.claude\projects\C--Users-samik-Documents-code-gnn-simulati      
  on-slotting-optimization-gnn-slotting-optimization\892b6815-25a9-4776-ba5a-a79911f8d43b.jsonl 