### 
# Tweedie with Covariates
###
module tw
using JuMP, Gurobi, MLBase

###
#Solves QP for optimal weights
#assuming bound on h : norm(h)^2 <= hmax dualized to lam1
#assuming bound on (a,b) : a^2 + b^2 <= bmax^2 dualized to lam 2
#has been dualized into parameter lambda.  
#k is a two parameter kernel: k(y1, y2)
###
#quadratic portion of obj.
function form_omega(ys, ws, k, k1)
	const n = length(ys)
	out = zeros(2n + 2, 2n + 2)
	for ix = 1:n
		KKei = vcat(1, ys[ix], 
					[ k(ys[jx], ys[ix]) for jx = 1:n], 
					[k1(ys[jx], ys[ix]) for jx = 1:n]
					)
		out += ws[ix] * KKei * KKei'
	end
	return out
end

#Hnorm portion of the regularizer  
function form_graham(ys, k, k1, k12, offset)
	const n = length(ys)
	K   = [  k(ys[ix], ys[jx]) for ix = 1:n, jx=1:n]
	K1  = [ k1(ys[ix], ys[jx]) for ix = 1:n, jx=1:n]
	K12 = [k12(ys[ix], ys[jx]) for ix = 1:n, jx=1:n]
	out = 	[K  K1';
	 		 K1  K12]
	out += offset * eye(2n)
	return out
end

function form_lincoeff(ys, ws, k1, k12)
	const n = length(ys)
	K1_T =  [ k1(ys[jx], ys[ix]) for ix = 1:n, jx=1:n]
	K12_T = [k12(ys[jx], ys[ix]) for ix = 1:n, jx=1:n]
	return 2 * vcat(0, 1, K1_T * ws, K12_T * ws)
end

#col_vals is the vector (b, a, alphas, betas)
function eval_score(col_vals, ys, new_y, k, k1)
	const n = length(ys)
	out = col_vals[1] + col_vals[2] * new_y 
	out += sum(col_vals[2 + ix] * k(ys[ix], new_y) for ix = 1:n)
	out += sum(col_vals[2 + n + ix] * k1(ys[ix], new_y) for ix = 1:n)
	return out	
end


#Call to Gurobi to do the lifting in the lazy way. 
#A smarter solution would combine proj gradient descent with a sweeping alg. 
#lam 1 is hnorm regularizer norm_H(h)^2 
#lam 2 is two norm regularize a^2 + b^2 
function fit_cond_score(ys, ws, k, k1, k12, lam1::Float64, lam2::Float64; 
					force_inc = true, output=false, TOL=1e-5)
	const n = length(ys)
	@assert n >=2 "Require at least 2 data points"

	omega = form_omega(ys, ws, k, k1)
	b = form_lincoeff(ys, ws, k1, k12)
	graham = form_graham(ys, k, k1, k12, TOL)

	m = Model(solver=GurobiSolver(OutputFlag=Int(output)))
	@variable(m, alphas[1:n])
	@variable(m, betas[1:n])
	@variable(m, null_coef[1:2])  #null[1] + null[2] * Y
	col_var = vcat(null_coef, alphas, betas)

	#Monotonicity constraints
	if force_inc
		indx = sortperm(ys)
		for ix = 2:n
			vs_ix = null_coef[2] * ys[indx[ix]] + 
					sum(alphas[jx] * k(ys[jx], ys[indx[ix]]) for jx = 1:n) + 
					sum(betas[jx] * k1(ys[jx], ys[indx[ix]]) for jx = 1:n) 
			vs_p = null_coef[2] * ys[indx[ix - 1]] + 
					sum(alphas[jx] * k(ys[jx], ys[indx[ix - 1]]) for jx = 1:n) + 
					sum(betas[jx] * k1(ys[jx], ys[indx[ix - 1]]) for jx = 1:n) 
			@constraint(m, vs_p <= vs_ix)
		end
	end

	#objective 
	obj = col_var' * omega * col_var + b' * col_var
	@objective(m, Min, obj + 
						lam1 * col_var[3:end]' * graham * col_var[3:end] + 
						lam2 * null_coef' * null_coef)
	status = solve(m)
	#@assert status== :Optimal "Error solving: $status"
	#println(status)

	col_vals = getvalue(col_var)
	vals = [col_vals[1] + col_vals[2] * ys[ix] + 
			sum(col_vals[2 + jx] * k(ys[jx], ys[ix]) for jx = 1:n) + 
			sum(col_vals[2 + n + jx] * k1(ys[jx], ys[ix]) for jx = 1:n) 
			for ix = 1:n]

	return vals, col_vals
end

#Neglect monotonicity and solve a linear system directly. 
#lam1 is hnorm regularizer
#lam2 is null_space regularizer
function fit_cond_score_ls(ys, ws, k, k1, k12, lam1::Float64, lam2::Float64; TOL=1e-5)
	const n = length(ys)

	Q = form_omega(ys, ws, k, k1)
	b = form_lincoeff(ys, ws, k1, k12)
	graham_h = form_graham(ys, k, k1, k12, TOL)

	Q += [lam2 * ones(2, 2)  	zeros(2, 2n);
		  zeros(2n, 2) 			lam1 * graham_h]

	#min x' Q x + b'x ->  Qx  = - .5 * b 
	col_vals = \(Q, -.5 * b)

	vals = [col_vals[1] + col_vals[2] * ys[ix] + 
			sum(col_vals[2 + jx] * k(ys[jx], ys[ix]) for jx = 1:n) + 
			sum(col_vals[2 + n + jx] * k1(ys[jx], ys[ix]) for jx = 1:n) 
			for ix = 1:n]

	return vals, col_vals
end

#kw is now the kernel for the weights
#kw: R -> R
function fit_NW_ws(xs, ys, new_x, kw)
	const n = length(xs)
	#use the standard scaling for conenience.
	const h = n^(-1/6)
	ws = [kw(norm(xs[ix] - new_x)/h) for ix = 1:n] #extra h scaling drops out
	ws /= sum(ws)
	return ws
end

#####
#A tuned model consists of
#	training data (xs, ys)
#	a procedure that given new_x -> weights	
#	Kernel parameters for the RKHS
#	lambdas 

# Evaluation entails
#	computing the new weights
#	solving the estimation problem for conditional score at new pt
#	outputting estimate	
######


####
#Assumes 
#   weight procedure is fixed: fit_ws(xs, ys, new_x) -> ws
#	Kernel parameters are fixed
#	lam2 fixed, small
#Only Tunes lam1
########
function tune_lam1(xs, ys, fit_ws, k, k1, k12, lam1_grid, lam2::Float64, TOL=1e-5)
	const num_lams = length(lam1_grid)
	const n = length(ys)
	cv_vals = zeros(2, num_lams) #mean and std
	for (ilam, lam) in enumerate(lam1_grid)
		#Nothing to learn in the estimation phase!
		function est_fun(indices)
			return xs[indices], ys[indices]
		end

		function eval_fun(fit, test_indices)
			(xs_train, ys_train) = fit
			#compute MSE on the trained values	
			MSE = 0.			
			for indx in test_indices
				#VG Check how test set is passed
				new_x, new_y = xs[indx], ys[indx]
				ws = fit_ws(xs_train, ys_train, new_x)
				vals, col_vals = fit_cond_score(ys_train, ws, k, k1, k12, lam, lam2, TOL=TOL)
				MSE += (new_y - (new_y + new_x[1]^2 * eval_score(col_vals, ys_train, new_y, k, k1)))^2
			end
			println(MSE)
			return MSE
		end

		#now evaluate the cross_val.  
		folds_results = cross_validate(est_fun, eval_fun, n, Kfold(n, 5))
		cv_vals[1, ilam] = mean(folds_results)
		cv_vals[2, ilam] = std(folds_results)
	end

	#For now, return everything
	cv_vals
end

#Some helper functions 
k_gauss(x, y, c)   =  exp(-c * (x - y)^2 )
k1_gauss(x, y, c)  = -2c * k_gauss(x, y, c) * (x-y)
k12_gauss(x, y, c) =  k_gauss(x, y, c) * (2c - 4c^2 *(x-y)^2)

end #ends module 