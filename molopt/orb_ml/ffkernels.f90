! MIT License
!
! Copyright (c) 2021-2022 Konstantin Karandashev
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in all
! copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
! SOFTWARE.


SUBROUTINE fgmo_sep_orb_sym_kernel_wders(num_scalar_reps,&
                    A_orb_atom_reps, A_orb_arep_rhos, A_orb_rhos,&
                    A_orb_atom_nums, A_orb_nums,&
                    A_max_num_orb_atom_reps, A_max_num_orbs, A_num_mols,&
                    sigmas, global_gauss, kernel_mat, num_kern_comps)
use ffkernels_module, only : scalar_rep_resc_orb_sep, self_cov_prods,&
        fgmo_sep_orb_kernel_element_wders, lin2gauss, el_norm_der_log
implicit none
integer, intent(in):: num_scalar_reps
integer, intent(in):: A_max_num_orb_atom_reps, A_max_num_orbs, A_num_mols
double precision, dimension(:,:,:,:), intent(in):: A_orb_atom_reps
double precision, dimension(:,:,:), intent(in):: A_orb_arep_rhos
double precision, dimension(:, :), intent(in):: A_orb_rhos
double precision, dimension(:), intent(in):: sigmas
integer, intent(in), dimension(:, :):: A_orb_atom_nums
integer, intent(in), dimension(:):: A_orb_nums
integer, intent(in):: num_kern_comps
logical, intent(in):: global_Gauss
double precision, dimension(:, :, :), intent(inout):: kernel_mat
double precision, dimension(:, :, :), allocatable:: A_orb_self_covs
double precision, dimension(:, :), allocatable:: A_self_covs
double precision, dimension(:, :, :, :), allocatable:: A_orb_atom_sreps
integer:: A_mol_counter1, A_mol_counter2

allocate(A_orb_atom_sreps(num_scalar_reps, A_max_num_orb_atom_reps, A_max_num_orbs, A_num_mols))

call scalar_rep_resc_orb_sep(A_orb_atom_reps, sigmas(2:num_scalar_reps+1)*2.0, num_scalar_reps, A_max_num_orb_atom_reps,&
        A_max_num_orbs, A_num_mols, A_orb_atom_sreps)

allocate(A_orb_self_covs(num_kern_comps, A_max_num_orbs, A_num_mols))
if (global_Gauss) then
    allocate(A_self_covs(num_kern_comps, A_num_mols))
    call self_cov_prods(num_scalar_reps, A_orb_atom_sreps,&
                    A_orb_arep_rhos, A_orb_atom_nums, A_orb_nums,&
                    A_max_num_orb_atom_reps, A_max_num_orbs, A_num_mols,&
                    A_orb_self_covs, num_kern_comps, A_orb_rhos, A_self_covs)
else
    call self_cov_prods(num_scalar_reps, A_orb_atom_sreps,&
                    A_orb_arep_rhos, A_orb_atom_nums, A_orb_nums,&
                    A_max_num_orb_atom_reps, A_max_num_orbs, A_num_mols,&
                    A_orb_self_covs, num_kern_comps)
endif

!$OMP PARALLEL DO PRIVATE(A_mol_counter1, A_mol_counter2) SCHEDULE(DYNAMIC)
do A_mol_counter1=1, A_num_mols
    do A_mol_counter2=1, A_mol_counter1
        if (global_Gauss) then
            call fgmo_sep_orb_kernel_element_wders(num_scalar_reps, A_orb_atom_sreps(:,:,:,A_mol_counter2),&
                A_orb_arep_rhos(:,:,A_mol_counter2), A_orb_rhos(:,A_mol_counter2),&
                A_orb_atom_nums(:, A_mol_counter2),A_orb_nums(A_mol_counter2),&
                A_max_num_orb_atom_reps, A_max_num_orbs,&
                A_orb_atom_sreps(:,:,:,A_mol_counter1), A_orb_arep_rhos(:,:,A_mol_counter1),&
                A_orb_rhos(:,A_mol_counter1), A_orb_atom_nums(:, A_mol_counter1), A_orb_nums(A_mol_counter1),&
                A_max_num_orb_atom_reps, A_max_num_orbs, sigmas, global_gauss,&
                kernel_mat(:, A_mol_counter2, A_mol_counter1), num_kern_comps,&
                A_orb_self_covs(:, :, A_mol_counter2), A_orb_self_covs(:, :, A_mol_counter1),&
                A_self_covs(:, A_mol_counter2), A_self_covs(:, A_mol_counter1))
        else
            call fgmo_sep_orb_kernel_element_wders(num_scalar_reps, A_orb_atom_sreps(:,:,:,A_mol_counter2),&
                A_orb_arep_rhos(:,:,A_mol_counter2), A_orb_rhos(:,A_mol_counter2),&
                A_orb_atom_nums(:, A_mol_counter2),A_orb_nums(A_mol_counter2),&
                A_max_num_orb_atom_reps, A_max_num_orbs,&
                A_orb_atom_sreps(:,:,:,A_mol_counter1), A_orb_arep_rhos(:,:,A_mol_counter1),&
                A_orb_rhos(:,A_mol_counter1), A_orb_atom_nums(:, A_mol_counter1), A_orb_nums(A_mol_counter1),&
                A_max_num_orb_atom_reps, A_max_num_orbs, sigmas, global_gauss,&
                kernel_mat(:, A_mol_counter2, A_mol_counter1), num_kern_comps,&
                A_orb_self_covs(:, :, A_mol_counter2), A_orb_self_covs(:, :, A_mol_counter1))
        endif
    enddo
enddo
!$OMP END PARALLEL DO

do A_mol_counter1=1, A_num_mols
    do A_mol_counter2=1, A_mol_counter1
        kernel_mat(:, A_mol_counter1, A_mol_counter2)=kernel_mat(:, A_mol_counter2, A_mol_counter1)
    enddo
enddo

END SUBROUTINE fgmo_sep_orb_sym_kernel_wders


SUBROUTINE fgmo_sep_orb_kernel_wders(num_scalar_reps,&
                    A_orb_atom_reps, A_orb_arep_rhos, A_orb_rhos,&
                    A_orb_atom_nums, A_orb_nums,&
                    A_max_num_orb_atom_reps, A_max_num_orbs, A_num_mols,&
                    B_orb_atom_reps, B_orb_arep_rhos, B_orb_rhos,&
                    B_orb_atom_nums, B_orb_nums,&
                    B_max_num_orb_atom_reps, B_max_num_orbs, B_num_mols,&
                    sigmas, global_gauss, kernel_mat, num_kern_comps)
use ffkernels_module, only : scalar_rep_resc_orb_sep, self_cov_prods,&
        fgmo_sep_orb_kernel_element_wders, lin2gauss, el_norm_der_log
implicit none
integer, intent(in):: num_scalar_reps
integer, intent(in):: A_max_num_orb_atom_reps, A_max_num_orbs, A_num_mols,&
                      B_max_num_orb_atom_reps, B_max_num_orbs, B_num_mols
double precision, dimension(:,:,:,:), intent(in):: A_orb_atom_reps, B_orb_atom_reps
double precision, dimension(:,:,:), intent(in):: A_orb_arep_rhos, B_orb_arep_rhos
double precision, dimension(:, :), intent(in):: A_orb_rhos, B_orb_rhos
double precision, dimension(:), intent(in):: sigmas
integer, intent(in), dimension(:, :):: A_orb_atom_nums, B_orb_atom_nums
integer, intent(in), dimension(:):: A_orb_nums, B_orb_nums
integer, intent(in):: num_kern_comps
logical, intent(in):: global_gauss
double precision, dimension(:, :, :), intent(inout):: kernel_mat
double precision, dimension(:, :, :), allocatable:: A_orb_self_covs, B_orb_self_covs
double precision, dimension(:, :), allocatable:: A_self_covs, B_self_covs
double precision, dimension(:, :, :, :), allocatable:: A_orb_atom_sreps, B_orb_atom_sreps
integer:: A_mol_counter, B_mol_counter

allocate(A_orb_atom_sreps(num_scalar_reps, A_max_num_orb_atom_reps, A_max_num_orbs, A_num_mols),&
        B_orb_atom_sreps(num_scalar_reps, B_max_num_orb_atom_reps, B_max_num_orbs, B_num_mols),&
        A_orb_self_covs(num_kern_comps, A_max_num_orbs, A_num_mols),&
        B_orb_self_covs(num_kern_comps, B_max_num_orbs, B_num_mols))

call scalar_rep_resc_orb_sep(A_orb_atom_reps, sigmas(2:num_scalar_reps+1)*2.0, num_scalar_reps, A_max_num_orb_atom_reps,&
        A_max_num_orbs, A_num_mols, A_orb_atom_sreps)
call scalar_rep_resc_orb_sep(B_orb_atom_reps, sigmas(2:num_scalar_reps+1)*2.0, num_scalar_reps, B_max_num_orb_atom_reps,&
        B_max_num_orbs, B_num_mols, B_orb_atom_sreps)

if (global_gauss) then
    allocate(A_self_covs(num_kern_comps, A_num_mols), B_self_covs(num_kern_comps, B_num_mols))
    call self_cov_prods(num_scalar_reps, A_orb_atom_sreps,&
                    A_orb_arep_rhos, A_orb_atom_nums, A_orb_nums,&
                    A_max_num_orb_atom_reps, A_max_num_orbs, A_num_mols,&
                    A_orb_self_covs, num_kern_comps, A_orb_rhos, A_self_covs)
    call self_cov_prods(num_scalar_reps, B_orb_atom_sreps,&
                    B_orb_arep_rhos, B_orb_atom_nums, B_orb_nums,&
                    B_max_num_orb_atom_reps, B_max_num_orbs, B_num_mols,&
                    B_orb_self_covs, num_kern_comps, B_orb_rhos, B_self_covs)
else
    call self_cov_prods(num_scalar_reps, A_orb_atom_sreps,&
                    A_orb_arep_rhos, A_orb_atom_nums, A_orb_nums,&
                    A_max_num_orb_atom_reps, A_max_num_orbs, A_num_mols,&
                    A_orb_self_covs, num_kern_comps)
    call self_cov_prods(num_scalar_reps, B_orb_atom_sreps,&
                    B_orb_arep_rhos, B_orb_atom_nums, B_orb_nums,&
                    B_max_num_orb_atom_reps, B_max_num_orbs, B_num_mols,&
                    B_orb_self_covs, num_kern_comps)

endif

!$OMP PARALLEL DO PRIVATE(A_mol_counter, B_mol_counter) SCHEDULE(DYNAMIC)
do A_mol_counter=1, A_num_mols
    do B_mol_counter=1, B_num_mols
        if (global_Gauss) then
            call fgmo_sep_orb_kernel_element_wders(num_scalar_reps, A_orb_atom_sreps(:,:,:,A_mol_counter),&
                A_orb_arep_rhos(:,:,A_mol_counter), A_orb_rhos(:,A_mol_counter),&
                A_orb_atom_nums(:, A_mol_counter), A_orb_nums(A_mol_counter),&
                A_max_num_orb_atom_reps, A_max_num_orbs,&
                B_orb_atom_sreps(:,:,:,B_mol_counter), B_orb_arep_rhos(:,:,B_mol_counter),&
                B_orb_rhos(:,B_mol_counter),&
                B_orb_atom_nums(:, B_mol_counter), B_orb_nums(B_mol_counter),&
                B_max_num_orb_atom_reps, B_max_num_orbs, sigmas, global_gauss,&
                kernel_mat(:, B_mol_counter, A_mol_counter), num_kern_comps,&
                A_orb_self_covs(:, :, A_mol_counter), B_orb_self_covs(:, :, B_mol_counter),&
                A_self_covs(:, A_mol_counter), B_self_covs(:, B_mol_counter))
        else
            call fgmo_sep_orb_kernel_element_wders(num_scalar_reps, A_orb_atom_sreps(:,:,:,A_mol_counter),&
                A_orb_arep_rhos(:,:,A_mol_counter), A_orb_rhos(:,A_mol_counter),&
                A_orb_atom_nums(:, A_mol_counter), A_orb_nums(A_mol_counter),&
                A_max_num_orb_atom_reps, A_max_num_orbs,&
                B_orb_atom_sreps(:,:,:,B_mol_counter), B_orb_arep_rhos(:,:,B_mol_counter),&
                B_orb_rhos(:,B_mol_counter),&
                B_orb_atom_nums(:, B_mol_counter), B_orb_nums(B_mol_counter),&
                B_max_num_orb_atom_reps, B_max_num_orbs, sigmas, global_gauss,&
                kernel_mat(:, B_mol_counter, A_mol_counter), num_kern_comps,&
                A_orb_self_covs(:, :, A_mol_counter), B_orb_self_covs(:, :, B_mol_counter))
        endif
    enddo
enddo
!$OMP END PARALLEL DO

END SUBROUTINE fgmo_sep_orb_kernel_wders


