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


! Deleting linear dependent entries.
SUBROUTINE flinear_dependent_entries(sym_kernel_mat, orthonormalized_vectors, num_elements,&
                      residue_tol_coeff, lambda_val, ascending_residue_order, indices_to_ignore)
implicit none
integer, intent(in):: num_elements
logical, intent(in):: ascending_residue_order
double precision, intent(in), dimension(:, :):: sym_kernel_mat
double precision, intent(inout), dimension(:, :):: orthonormalized_vectors
double precision, intent(in):: residue_tol_coeff, lambda_val
integer, intent(inout), dimension(:):: indices_to_ignore
double precision, dimension(num_elements, num_elements):: temp_kernel_mat
double precision, dimension(num_elements):: sqnorm_residue, sqnorm_true
logical, dimension(num_elements):: to_ignore, considered
integer:: i, j, cur_orth_id
double precision:: cur_norm_ratio, cur_product, cur_norm
integer, dimension(num_elements):: orthonorm_order
double precision:: proc_extr_residue, global_extr_residue
integer:: proc_extr_res_id, global_extr_res_id
logical:: proc_extr_res_init, global_extr_res_init
logical:: matrix_unstable

    matrix_unstable=.False.

    global_extr_res_init=.False.

    orthonormalized_vectors=0.0
    considered=.False.

    orthonorm_order=0

!$OMP PARALLEL PRIVATE(i, cur_norm_ratio, proc_extr_res_id, proc_extr_res_init, proc_extr_residue)

    proc_extr_res_init=.False.

!$OMP DO
    do i=1, num_elements
        sqnorm_true(i)=sym_kernel_mat(i, i)
        sqnorm_residue(i)=sqnorm_true(i)+lambda_val
        cur_norm_ratio=sqnorm_residue(i)/sqnorm_true(i)
        call compare_replace(ascending_residue_order,&
            proc_extr_residue, proc_extr_res_id, proc_extr_res_init,&
            cur_norm_ratio, i)
    enddo
!$OMP END DO

    if (proc_extr_res_init) then
!$OMP CRITICAL
        call compare_replace(ascending_residue_order,&
            global_extr_residue, global_extr_res_id, global_extr_res_init,&
            proc_extr_residue, proc_extr_res_id)
!$OMP END CRITICAL
    endif

!$OMP END PARALLEL

    cur_orth_id=1

    do
        orthonormalized_vectors(cur_orth_id, global_extr_res_id)=1.0
        orthonorm_order(cur_orth_id)=global_extr_res_id
        temp_kernel_mat(cur_orth_id, :)=sym_kernel_mat(global_extr_res_id, :)
        considered(global_extr_res_id)=.True.
        ! Normalize the vector.
        cur_norm=sqrt(sqnorm_residue(global_extr_res_id))
        orthonormalized_vectors(1:cur_orth_id, global_extr_res_id)=&
                orthonormalized_vectors(1:cur_orth_id, global_extr_res_id)/cur_norm
        ! Subtract projections of the normalized vector from all currently not orthonormalized vectors.
        ! Also check that their residue is above the corresponding threshold.

        global_extr_res_init=.False.
!$OMP PARALLEL PRIVATE(i, cur_product, proc_extr_res_id, proc_extr_res_init, proc_extr_residue, cur_norm_ratio)

        proc_extr_res_init=.False.

!$OMP DO
        do i=1, num_elements
            if (.not.considered(i)) then
                cur_product=dot_product(orthonormalized_vectors(1:cur_orth_id, global_extr_res_id),&
                                        temp_kernel_mat(1:cur_orth_id, i))
                sqnorm_residue(i)=sqnorm_residue(i)-cur_product**2
                cur_norm_ratio=sqnorm_residue(i)/sqnorm_true(i)
                if (cur_norm_ratio<residue_tol_coeff) then
                    considered(i)=.True.
                    if (cur_norm_ratio<-residue_tol_coeff) then
!$OMP CRITICAL
                        matrix_unstable=.True.
!$OMP END CRITICAL
                    endif
                else
                    orthonormalized_vectors(1:cur_orth_id,i)=&
                        orthonormalized_vectors(1:cur_orth_id,i)-cur_product*&
                        orthonormalized_vectors(1:cur_orth_id, global_extr_res_id)
                    call compare_replace(ascending_residue_order,&
                        proc_extr_residue, proc_extr_res_id, proc_extr_res_init,&
                        cur_norm_ratio, i)
                endif
            endif
        enddo
!$OMP END DO

        if (proc_extr_res_init) then
!$OMP CRITICAL
            call compare_replace(ascending_residue_order,&
                    global_extr_residue, global_extr_res_id, global_extr_res_init,&
                    proc_extr_residue, proc_extr_res_id)
!$OMP END CRITICAL
        endif

!$OMP END PARALLEL

        if (matrix_unstable) then
            indices_to_ignore(1)=-2
            return
        endif
        if (.not.global_extr_res_init) exit
        cur_orth_id=cur_orth_id+1
    enddo


    to_ignore=.True.
    temp_kernel_mat=0.0
!$OMP PARALLEL DO PRIVATE(i)
    do i=1, cur_orth_id
        to_ignore(orthonorm_order(i))=.False.
        temp_kernel_mat(orthonorm_order(i), :)=orthonormalized_vectors(i, :)
    enddo
!$OMP END PARALLEL DO

!$OMP PARALLEL DO PRIVATE(i)
    do i=1, num_elements
        if (to_ignore(i)) temp_kernel_mat(:, i)=0.0
    enddo
!$OMP END PARALLEL DO

    orthonormalized_vectors(:, :)=temp_kernel_mat(:, :)

    indices_to_ignore=-1
    ! Create a list with ignored indices.
    j=1
    do i=1, num_elements
        if (to_ignore(i)) then
!            print *, 'Skipped:', i-1, 'residue:', sqnorm_residue(i) ! Subtracting 1 because later used in python scripts.
            indices_to_ignore(j)=i-1 ! Subtracting 1 because later used in python scripts.
            j=j+1
        endif
    enddo
END SUBROUTINE

PURE SUBROUTINE compare_replace(leave_smaller_val, init_val, init_id, init_initialized, new_val, new_id)
logical, intent(in):: leave_smaller_val
double precision, intent(inout):: init_val
integer, intent(inout):: init_id
logical, intent(inout):: init_initialized
double precision, intent(in):: new_val
integer, intent(in):: new_id
logical:: favorable

    if (leave_smaller_val) then
        favorable=(init_val>new_val)
    else
        favorable=(init_val<new_val)
    endif
    if (favorable.or.(.not.init_initialized)) then
        init_initialized=.True.
        init_val=new_val
        init_id=new_id
    endif

END SUBROUTINE

