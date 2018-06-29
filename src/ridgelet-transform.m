%定数
%I = 601;
%J = 601;
%Delta_a = 60.0 / (I - 1);
%Delta_b = 60.0 / (J - 1);
%a = linspace(-30, 30, I);
%b = linspace(-30, 30, J);
%
%N = 201;
%Delta_x = 2.0 / (N - 1);
%x = linspace(-1, 1, N);

%学習すべき関数
%function r = f(x)
%	r = sin(2*pi*x);
%	#r = 0.7 .* sin(2 * pi * x) + 0.3 .* cos(6 * pi * x);
%	#r = 0.7 .* sin(2 * pi * x) - 0.3 .* sin(6 * pi * x);
%	#r = cos(3 * pi * x) .* exp(-4 * x.^2);
%endfunction

%リッジレット関数
%function r = psi(x)
%	#r = (1 .- sqrt(2) .* x .* dawson(x ./ sqrt(2))) ./ (sqrt(2) * pi^2);
%	r = (2 .* x .* (x.^2 - 3) .* dawson(x ./ sqrt(2)) .- sqrt(2) .* (x.^2 - 2)) ./ pi^2;
%endfunction

source('./temp/function.m')

%活性化関数
function r = eta(x)
	r = exp(-x .^ 2 ./ 2);
endfunction

%profile on

%ベクトルを複製して3次元配列に拡張する
disp('duplicating vectors...');
a3d = repmat(reshape(a, [I 1 1]), [1, J, N]); %(a_i)_ijk
b3d = repmat(reshape(b, [1 J 1]), [I, 1, N]); %(b_j)_ijk
x3d = repmat(reshape(x, [1 1 N]), [I, J, 1]); %(x_k)_ijk
disp('done.');

%リッジレット変換の数値計算
disp('calculating ridgelet transformation...');
T = sum(f(x3d) .* psi(a3d .* x3d .- b3d) * Delta_x, 3);
disp('done.');

filename = './output/numerical-ridgelet.txt';
disp(cstrcat('writing to ', filename))
dlmwrite(filename, T, 'delimiter', ' ', 'newline', '\n');
disp('done.');

%双対リッジレット変換の数値計算
disp('calculating dual ridgelet transform...');
F = vec(sum(sum(T .* eta(a3d .* x3d .- b3d) * (Delta_a * Delta_b), 2), 1));
disp('done.');

filename = './output/numerical-dual-ridgelet.txt';
disp(cstrcat('writing to ', filename));
dlmwrite(filename, [vec(x) F], 'delimiter', ' ', 'newline', '\n');
disp('done.');

%profile off
%profshow(profile("info"))

