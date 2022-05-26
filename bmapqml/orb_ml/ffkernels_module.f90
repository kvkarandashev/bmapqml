MODULE ffkernels_module
implicit none
public

contains


SUBROUTINE scalar_rep_resc_orb_sep(array_in, width_params, dim1, dim2, dim3, dim4, array_out)
implicit none
integer, intent(in):: dim1, dim2, dim3, dim4
double precision, dimension(dim1, dim2, dim3, dim4), intent(in):: array_in
double precision, dimension(dim1, dim2, dim3, dim4), intent(inout):: array_out
double precision, dimension(dim1), intent(in):: width_params
integer:: i2, i3, i4

!$OMP PARALLEL DO
do i4=1, dim4
    do i3=1, dim3
        do i2=1, dim2
            array_out(:, i2, i3, i4)=array_in(:, i2, i3, i4)/width_params
        enddo
    enddo
enddo
!$OMP END PARALLEL DO


END SUBROUTINE scalar_rep_resc_orb_sep

PURE SUBROUTINE orb_orb_lin_cov_wders(num_scalar_reps,&
            A_orb_atom_sreps, A_orb_arep_rhos, A_orb_atom_num,&
            A_max_num_orb_atom_reps,&
            B_orb_atom_sreps, B_orb_arep_rhos, B_orb_atom_num,&
            B_max_num_orb_atom_reps,&
            orb_cov_components, num_kern_comps)
implicit none
integer, intent(in):: num_scalar_reps, A_max_num_orb_atom_reps,&
                B_max_num_orb_atom_reps, A_orb_atom_num, B_orb_atom_num,&
                num_kern_comps
double precision, dimension(num_scalar_reps, A_max_num_orb_atom_reps),&
        intent(in):: A_orb_atom_sreps
double precision, dimension(num_scalar_reps, B_max_num_orb_atom_reps),&
        intent(in):: B_orb_atom_sreps
double precision, dimension(A_max_num_orb_atom_reps), intent(in):: A_orb_arep_rhos
double precision, dimension(B_max_num_orb_atom_reps), intent(in):: B_orb_arep_rhos
double precision, dimension(num_kern_comps), intent(inout):: orb_cov_components
integer:: A_arep_id, B_arep_id
double precision:: exp_fac
double precision, dimension(num_scalar_reps):: sqdiff_vec
integer:: lin_kern_size

if (num_kern_comps==1) then
    lin_kern_size=1
else
    lin_kern_size=num_scalar_reps+1
endif


orb_cov_components=0.0
do A_arep_id=1, A_orb_atom_num
    do B_arep_id=1, B_orb_atom_num
        sqdiff_vec=(A_orb_atom_sreps(:, A_arep_id)-B_orb_atom_sreps(:, B_arep_id))**2
        exp_fac=exp(-sum(sqdiff_vec))*A_orb_arep_rhos(A_arep_id)&
                            *B_orb_arep_rhos(B_arep_id)
        orb_cov_components(1)=orb_cov_components(1)+exp_fac
        if (num_kern_comps/=1) then
            orb_cov_components(2:lin_kern_size)=orb_cov_components(2:lin_kern_size)&
                    -exp_fac*sqdiff_vec
        endif
    enddo
enddo

END SUBROUTINE orb_orb_lin_cov_wders

PURE SUBROUTINE el_norm_der_log(cov_components, A_self_covs, B_self_covs, num_kern_comps)
integer, intent(in):: num_kern_comps
double precision, dimension(num_kern_comps), intent(inout):: cov_components
double precision, dimension(num_kern_comps), intent(in):: A_self_covs, B_self_covs

    cov_components(:)=cov_components(:)/A_self_covs(1)/B_self_covs(1)
    if (num_kern_comps/=1) & 
        cov_components(2:num_kern_comps-1)=cov_components(2:num_kern_comps-1)&
            -(A_self_covs(2:num_kern_comps-1)+B_self_covs(2:num_kern_comps-1))/2.0*cov_components(1)

END SUBROUTINE el_norm_der_log


PURE SUBROUTINE orb_self_lin_cov_wders(num_scalar_reps,&
            A_orb_atom_sreps, A_orb_arep_rhos, A_orb_atom_num,&
            A_max_num_orb_atom_reps, orb_cov_components, num_kern_comps)
implicit none
integer, intent(in):: num_scalar_reps, A_max_num_orb_atom_reps,&
                A_orb_atom_num, num_kern_comps
double precision, dimension(num_scalar_reps, A_max_num_orb_atom_reps),&
        intent(in):: A_orb_atom_sreps
double precision, dimension(A_max_num_orb_atom_reps), intent(in):: A_orb_arep_rhos
double precision, dimension(num_kern_comps), intent(inout):: orb_cov_components

    call orb_orb_lin_cov_wders(num_scalar_reps,&
            A_orb_atom_sreps, A_orb_arep_rhos, A_orb_atom_num,&
            A_max_num_orb_atom_reps,&
            A_orb_atom_sreps, A_orb_arep_rhos, A_orb_atom_num,&
            A_max_num_orb_atom_reps,&
            orb_cov_components, num_kern_comps)


END SUBROUTINE orb_self_lin_cov_wders

PURE SUBROUTINE lin2gauss(converted_comps, inv_sq_sigma, num_kern_comps)
integer, intent(in):: num_kern_comps
double precision, intent(inout), dimension(num_kern_comps):: converted_comps
double precision, intent(in):: inv_sq_sigma

    if (num_kern_comps/=1) then
        converted_comps(3:num_kern_comps)=converted_comps(2:num_kern_comps-1)*inv_sq_sigma
        converted_comps(2)=converted_comps(1)-1.0
    endif
    converted_comps(1)=exp(-inv_sq_sigma*(1.0-converted_comps(1)))
    if (num_kern_comps/=1) converted_comps(2:num_kern_comps)=&
            converted_comps(2:num_kern_comps)*converted_comps(1)

END SUBROUTINE lin2gauss

PURE SUBROUTINE flmo_sep_orb_kernel_element_wders(num_scalar_reps,&
            A_orb_atom_sreps, A_orb_arep_rhos, A_orb_rhos, A_orb_atom_nums, A_orb_num,&
            A_max_num_orb_atom_reps, A_max_num_orbs,&
            B_orb_atom_sreps, B_orb_arep_rhos,&
            B_orb_rhos, B_orb_atom_nums, B_orb_num,&
            B_max_num_orb_atom_reps, B_max_num_orbs, global_gauss,&
            kernel_elements, num_kern_comps, A_orb_self_covs, B_orb_self_covs, inv_sq_sigma)
implicit none
integer, intent(in):: num_scalar_reps, A_max_num_orb_atom_reps, A_max_num_orbs,&
                        num_kern_comps, B_max_num_orb_atom_reps, B_max_num_orbs
integer, intent(in):: A_orb_num, B_orb_num
double precision, intent(in), dimension(num_scalar_reps, A_max_num_orb_atom_reps,&
                A_max_num_orbs):: A_orb_atom_sreps
double precision, intent(in), dimension(num_scalar_reps, B_max_num_orb_atom_reps,&
                B_max_num_orbs):: B_orb_atom_sreps
double precision, intent(in), dimension(A_max_num_orb_atom_reps,&
            A_max_num_orbs):: A_orb_arep_rhos
double precision, intent(in), dimension(B_max_num_orb_atom_reps,&
            B_max_num_orbs):: B_orb_arep_rhos
double precision, intent(in), dimension(A_max_num_orbs):: A_orb_rhos
double precision, intent(in), dimension(B_max_num_orbs):: B_orb_rhos
integer, dimension(A_max_num_orbs), intent(in):: A_orb_atom_nums
integer, dimension(B_max_num_orbs), intent(in):: B_orb_atom_nums
double precision, intent(in), optional:: inv_sq_sigma
logical, intent(in):: global_gauss
double precision, dimension(num_kern_comps), intent(inout):: kernel_elements
double precision, dimension(num_kern_comps, A_max_num_orbs), intent(in):: A_orb_self_covs
double precision, dimension(num_kern_comps, B_max_num_orbs), intent(in):: B_orb_self_covs
integer:: A_orb_id, B_orb_id
double precision, dimension(num_kern_comps):: orb_cov_components

kernel_elements=0.0

do A_orb_id = 1, A_orb_num
    do B_orb_id = 1, B_orb_num
        call orb_orb_lin_cov_wders(num_scalar_reps,&
            A_orb_atom_sreps(:, :, A_orb_id), A_orb_arep_rhos(:, A_orb_id),&
            A_orb_atom_nums(A_orb_id), A_max_num_orb_atom_reps,&
            B_orb_atom_sreps(:, :, B_orb_id), B_orb_arep_rhos(:, B_orb_id),&
            B_orb_atom_nums(B_orb_id), B_max_num_orb_atom_reps,&
            orb_cov_components, num_kern_comps)

        call el_norm_der_log(orb_cov_components, A_orb_self_covs(:, A_orb_id),&
                                    B_orb_self_covs(:, B_orb_id), num_kern_comps)

        if (.not.global_gauss) call lin2gauss(orb_cov_components, inv_sq_sigma, num_kern_comps)

        kernel_elements=kernel_elements+orb_cov_components*A_orb_rhos(A_orb_id)*B_orb_rhos(B_orb_id)
    enddo
enddo


END SUBROUTINE flmo_sep_orb_kernel_element_wders


PURE SUBROUTINE fgmo_sep_orb_kernel_element_wders(num_scalar_reps,&
            A_orb_atom_sreps, A_orb_arep_rhos, A_orb_rhos, A_orb_atom_nums, A_orb_num,&
            A_max_num_orb_atom_reps, A_max_num_orbs,&
            B_orb_atom_sreps, B_orb_arep_rhos,&
            B_orb_rhos, B_orb_atom_nums, B_orb_num,&
            B_max_num_orb_atom_reps, B_max_num_orbs, sigmas, global_gauss,&
            kernel_elements, num_kern_comps, A_orb_self_covs, B_orb_self_covs,&
            A_self_covs, B_self_covs)
implicit none
integer, intent(in):: num_scalar_reps, A_max_num_orb_atom_reps, A_max_num_orbs,&
                        num_kern_comps, B_max_num_orb_atom_reps, B_max_num_orbs
integer, intent(in):: A_orb_num, B_orb_num
double precision, intent(in), dimension(num_scalar_reps, A_max_num_orb_atom_reps,&
                A_max_num_orbs):: A_orb_atom_sreps
double precision, intent(in), dimension(num_scalar_reps, B_max_num_orb_atom_reps,&
                B_max_num_orbs):: B_orb_atom_sreps
double precision, intent(in), dimension(A_max_num_orb_atom_reps,&
            A_max_num_orbs):: A_orb_arep_rhos
double precision, intent(in), dimension(B_max_num_orb_atom_reps,&
            B_max_num_orbs):: B_orb_arep_rhos
double precision, intent(in), dimension(A_max_num_orbs):: A_orb_rhos
double precision, intent(in), dimension(B_max_num_orbs):: B_orb_rhos
integer, dimension(A_max_num_orbs), intent(in):: A_orb_atom_nums
integer, dimension(B_max_num_orbs), intent(in):: B_orb_atom_nums
double precision, dimension(num_scalar_reps+1), intent(in):: sigmas
logical, intent(in):: global_gauss
double precision, dimension(num_kern_comps), intent(inout):: kernel_elements
double precision, dimension(num_kern_comps, A_max_num_orbs), intent(in):: A_orb_self_covs
double precision, dimension(num_kern_comps, B_max_num_orbs), intent(in):: B_orb_self_covs
double precision, dimension(num_kern_comps), intent(in), optional:: A_self_covs, B_self_covs
double precision:: inv_sq_sigma

inv_sq_sigma=1.0/sigmas(1)**2

call flmo_sep_orb_kernel_element_wders(num_scalar_reps,&
            A_orb_atom_sreps, A_orb_arep_rhos, A_orb_rhos, A_orb_atom_nums, A_orb_num,&
            A_max_num_orb_atom_reps, A_max_num_orbs,&
            B_orb_atom_sreps, B_orb_arep_rhos,&
            B_orb_rhos, B_orb_atom_nums, B_orb_num,&
            B_max_num_orb_atom_reps, B_max_num_orbs, global_gauss,&
            kernel_elements, num_kern_comps, A_orb_self_covs, B_orb_self_covs,&
            inv_sq_sigma)

if (global_gauss) then
    call el_norm_der_log(kernel_elements, A_self_covs, B_self_covs, num_kern_comps)
    call lin2gauss(kernel_elements, inv_sq_sigma, num_kern_comps)
endif
if (num_kern_comps /= 1) then
    kernel_elements(2:num_kern_comps)=-kernel_elements(2:num_kern_comps)/sigmas*2
    kernel_elements(2)=kernel_elements(2)*inv_sq_sigma
endif


END SUBROUTINE fgmo_sep_orb_kernel_element_wders


SUBROUTINE self_cov_prods(num_scalar_reps, A_orb_atom_sreps,&
                    A_orb_arep_rhos, A_orb_atom_nums, A_orb_nums,&
                    A_max_num_orb_atom_reps, A_max_num_orbs, A_num_mols,&
                    A_orb_self_covs, num_kern_comps, A_orb_rhos, A_self_covs)
implicit none
integer, intent(in):: num_scalar_reps, A_max_num_orb_atom_reps, A_max_num_orbs, A_num_mols, num_kern_comps
double precision, dimension(num_scalar_reps, A_max_num_orb_atom_reps,&
                                A_max_num_orbs, A_num_mols), intent(in):: A_orb_atom_sreps
double precision, dimension(A_max_num_orb_atom_reps,&
                                A_max_num_orbs, A_num_mols), intent(in):: A_orb_arep_rhos
integer, dimension(A_max_num_orbs, A_num_mols), intent(in):: A_orb_atom_nums
integer, dimension(A_num_mols), intent(in):: A_orb_nums
double precision, dimension(num_kern_comps, A_max_num_orbs, A_num_mols), intent(inout):: A_orb_self_covs
double precision, dimension(A_max_num_orbs, A_num_mols), intent(in), optional:: A_orb_rhos
double precision, dimension(num_kern_comps, A_num_mols), intent(inout), optional:: A_self_covs
integer:: mol_id, orb_id
integer:: der_upper_id
double precision:: dummy_inv_sq_sigma

    if (num_kern_comps/=1) der_upper_id=num_kern_comps-1

!$OMP PARALLEL DO PRIVATE(mol_id, orb_id) SCHEDULE(DYNAMIC)
    do mol_id=1, A_num_mols
        do orb_id=1, A_orb_nums(mol_id)
            call orb_self_lin_cov_wders(num_scalar_reps,&
            A_orb_atom_sreps(:, :, orb_id, mol_id), A_orb_arep_rhos(:, orb_id, mol_id),&
            A_orb_atom_nums(orb_id, mol_id), A_max_num_orb_atom_reps, A_orb_self_covs(:, orb_id, mol_id), num_kern_comps)
            if (num_kern_comps/=1) &
            A_orb_self_covs(2:der_upper_id, orb_id, mol_id)=A_orb_self_covs(2:der_upper_id, orb_id, mol_id)&
                                /A_orb_self_covs(1, orb_id, mol_id)
            A_orb_self_covs(1, orb_id, mol_id)=sqrt(A_orb_self_covs(1, orb_id, mol_id))
        enddo
        if (present(A_orb_rhos)) then
            call flmo_sep_orb_kernel_element_wders(num_scalar_reps,&
                A_orb_atom_sreps(:, :, :, mol_id), A_orb_arep_rhos(:, :, mol_id),&
                A_orb_rhos(:, mol_id), A_orb_atom_nums(:,mol_id), A_orb_nums(mol_id),&
                A_max_num_orb_atom_reps, A_max_num_orbs,&
                A_orb_atom_sreps(:, :, :, mol_id), A_orb_arep_rhos(:,:,mol_id),&
                A_orb_rhos(:,mol_id), A_orb_atom_nums(:,mol_id), A_orb_nums(mol_id),&
                A_max_num_orb_atom_reps, A_max_num_orbs, .TRUE.,&
                A_self_covs(:, mol_id), num_kern_comps,&
                A_orb_self_covs(:, :, mol_id), A_orb_self_covs(:, :, mol_id), dummy_inv_sq_sigma)
            if (num_kern_comps/=1) A_self_covs(2:der_upper_id, mol_id)=A_self_covs(2:der_upper_id, mol_id)/A_self_covs(1, mol_id)
            A_self_covs(1, mol_id)=sqrt(A_self_covs(1, mol_id))
        endif
    enddo
!$OMP END PARALLEL DO


END SUBROUTINE self_cov_prods


END MODULE ffkernels_module
