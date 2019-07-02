import netCDF4 as 
class Utils:
	def export_netcdf(self, filename, var_name):
		nc = n.Dataset(filename, 'r', format='NETCDF4_CLASSIC')
		var = nc.variables[var_name][:]
		return var

       def build_tensor(self, data_dir, file_start, file_end, n_files):
                """
                Builds dataset out of all climate variables of shape HxWxNxC, where:
                        N - #of datapoints (days)
                        H - latitude
                        W - longitude
                        C - #of channels (climate variables)

                :return dataset as a tensor
                """

                #all tensors is a list of tensor, where each tensor has data about one climate variable
                all_tensors = []
                count_files = 0
                for i, (key, val) in enumerate(clmt_vars.items()):
                        clmt_dir = val[0]
                        file_dir = data_dir + clmt_dir
                        #create tensor for one climate variable
                        tensors_per_clmt_var = []
                        #sort files in ascending order (based on the date)
                        filenames = os.listdir(file_dir)
                        filenames = sorted(filenames)
                        count_files += len(filenames)
                        for i in range(file_start, file_end + 1):
				filename = filenames[i]
                                raw_clmt_data = self.export_netcdf(file_dir + filename,key)
				#check if we need only partial data from the current file
				if i == file_end and file_end == n_files - 1:
					days_processed = train_len - raw_clmt_data.shape[0]
					#days_to_process = 
                                	raw_tsr = torch.tensor(raw_clmt_data[:ub, :, :], dtype=torch.float32)
                                	tensors_per_clmt_var.append(raw_tsr)
			
                        #concatenate tensors along the size dimension
                        concat_tsr = torch.cat(tensors_per_clmt_var, dim=0)
                        all_tensors.append(concat_tsr)
        		assert n_days_processed == concat_tsr.shape[0], "        
		res_tsr = torch.stack(all_tensors, dim=3)
                #permuate tensor for convenience
                res_tsr = res_tsr.permute(1, 2, 0, 3)#H x W x T x N
                return res_tsr
