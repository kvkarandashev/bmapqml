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

PURE SUBROUTINE fgaussian_kernel_matrix_element(A_vec, B_vec, inv_sigma_param, sigma_der_param,&
                    num_features, num_kern_comps, kernel_element)
implicit none
integer, intent(in):: num_features, num_kern_comps
double precision, intent(in):: inv_sigma_param, sigma_der_param
double precision, intent(inout), dimension(num_kern_comps):: kernel_element
double precision, intent(in), dimension(num_features):: A_vec, B_vec
double precision:: sqdist

    sqdist=sum((A_vec-B_vec)**2)
    kernel_element(1)=exp(-sqdist*inv_sigma_param)
    if (num_kern_comps /= 1) then
        kernel_element(2)=kernel_element(1)*sqdist*sigma_der_param
    endif

END SUBROUTINE

SUBROUTINE fgaussian_related_parameters(sigma, inv_sigma_param, sigma_der_param, num_kern_comps)
implicit none
double precision, intent(in):: sigma
integer, intent(in):: num_kern_comps
double precision, intent(inout):: inv_sigma_param, sigma_der_param

    inv_sigma_param=.5/sigma**2
    if (num_kern_comps /= 1) then
        sigma_der_param=sigma**(-3)
    endif

END SUBROUTINE

SUBROUTINE fgaussian_kernel_matrix(A, B, sigma, num_kern_comps, num_A, num_B, num_features, kernel_matrix)
implicit none
double precision, dimension(:, :), intent(in):: A, B
integer, intent(in):: num_A, num_B, num_features, num_kern_comps
double precision, intent(in):: sigma
double precision, dimension(:, :, :), intent(inout):: kernel_matrix
integer:: i_A, i_B
double precision:: inv_sigma_param, sigma_der_param

call fgaussian_related_parameters(sigma, inv_sigma_param, sigma_der_param, num_kern_comps)

!$OMP PARALLEL DO
    do i_A=1, num_A
        do i_B=1, num_B
            call fgaussian_kernel_matrix_element(A(:, i_A), B(:, i_B), inv_sigma_param, sigma_der_param,&
                    num_features, num_kern_comps, kernel_matrix(:, i_B, i_A))
        enddo
    enddo
!$OMP END PARALLEL DO

END SUBROUTINE


SUBROUTINE fgaussian_sym_kernel_matrix(A, sigma, num_kern_comps, num_A, num_features, kernel_matrix)
implicit none
double precision, dimension(:, :), intent(in):: A
double precision, intent(in):: sigma
integer, intent(in):: num_A, num_features, num_kern_comps
double precision, dimension(:, :, :), intent(inout):: kernel_matrix
integer:: i_A1, i_A2
double precision:: inv_sigma_param, sigma_der_param

    call fgaussian_related_parameters(sigma, inv_sigma_param, sigma_der_param, num_kern_comps)

!$OMP PARALLEL DO SCHEDULE(DYNAMIC)
    do i_A1=1, num_A
        do i_A2=1, i_A1
            call fgaussian_kernel_matrix_element(A(:, i_A1), A(:, i_A2), inv_sigma_param, sigma_der_param,&
                    num_features, num_kern_comps, kernel_matrix(:, i_A2, i_A1))
        enddo
    enddo
!$OMP END PARALLEL DO

!$OMP PARALLEL DO SCHEDULE(DYNAMIC)
    do i_A1=1, num_A
        do i_A2=1, i_A1
            kernel_matrix(:, i_A1, i_A2)=kernel_matrix(:, i_A2, i_A1)
        enddo
    enddo
!$OMP END PARALLEL DO


END SUBROUTINE

