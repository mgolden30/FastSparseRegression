function [cs, residuals] = SPRINT_minus( G )
  %{
  PURPOSE:
  Find approximate minima of L = |G*c|_2 / |c|_2 in nested sparse subspaces, such that L
  increases minimally at each stage.

  This variant of SPRINT starts with the full library and removes terms one at a time.
  See SPRINT_plus to add terms one at a time.


  INPUT:
  G - a matrix to look for sparse null vectors of



  OUTPUT:
  cs - columns of this matrix are the increasingly sparse approximate null
       vectors.
  residuals - vecnorm( G*cs )/sqrt(m);



  %}




  m = size(G,1);
  n = size(G,2);

  cs = 0;

  if m<n
    fprintf("error: matrix is underdetermined.");
    return;
  end

  %first rotate A so it is square and upper triangular
  [~, G] = qr(G);  G = G(1:n, 1:n);
  
  %keep a copy
  A0 = G; 

  cs = zeros(n,n);
  I  = ones(n,1); I = (I == 1); %logical vector indicating sparsity
  residuals = zeros(n,1);

  while( n > 0 )
    [U, S, V] = svd(G, 'econ'); 
    cs(I,n) = V(:,n);   %save out the smallest singular vector
    residuals(n) = S(n,n);

    if( n == 1 )
      break;
    end

    candidates = zeros(n,1);
    for i = 1:n
      a = G(:,i);
      alpha = 1/norm(a);
      w = alpha*U'*a;

      ws = [w(end-1), w(end)];

      s = diag(S); %turn singular vectors into array
      bounds = [s(end), s(end-1)];
      

      s1 = s(end);
      s2 = s(end-1);
      s  = s(1:end-2);
      w1 = w(end);
      w2 = w(end-1);
      w  = w(1:end-2);


      first_term  = @(sigma)  ( s1^2 - sigma^2 ) .* ( s2^2 - sigma^2 ) * alpha^2 / (s1^2 - s2^2);
      second_term = @(sigma)  - w1^2 * (s2^2 - sigma.^2 )/(s1^2 - s2^2);
      third_term  = @(sigma)  - w2^2 * (s1^2 - sigma.^2 )/(s1^2 - s2^2);
      fourth_term = @(sigma)  - sum( w.^2 ./ ( s.^2 - sigma.^2 ) ) * ( s1^2 - sigma.^2 ) .* ( s2^2 - sigma.^2 ) / (s1^2 - s2^2);
     
      r = s1/s2;
      first_term  = @(sigma)  ( r^2 - (sigma/s2)^2 ) .* ( s2^2 - sigma^2 ) * alpha^2 / (r^2 - 1);
      second_term = @(sigma)  - w1^2 * (1^2  - (sigma/s2).^2 )/(r^2 - 1);
      third_term  = @(sigma)  - w2^2 * (r^2 - (sigma/s2).^2 )/(r^2 - 1);
      fourth_term = @(sigma)  - sum( w.^2 ./ ( (s/s2).^2 - (sigma/s2).^2 ) ) * ( r^2 - (sigma/s2).^2 ) .* ( 1 - (sigma/s2).^2 ) / (r^2 - 1);
    
      f = @(sigma)  first_term(sigma) + second_term(sigma) + third_term(sigma) + fourth_term(sigma);

      maxit = 128;
      threshold = 1e-130;
      g = 0;

      for j = 1:maxit
        g  = sum(bounds)/2; %bisection guess
        fg = f(g);
        if(abs(fg) < threshold)
          break;
        end

        if (fg > 0  )
          bounds(1) = g;
        else
          bounds(2) = g;
        end
      end
      candidates(i) = g;
    end

    [~, i_min] = min( candidates );

    j = find(I);
    I(j(i_min)) = 0;
    G = A0(:, I);
    %size(G)
    n = n-1;
  end

  %rescale the residual
  residuals = residuals / sqrt(m);
end