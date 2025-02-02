% function windowed_data = applySineWindow(data, off, endVal, pow, c)
%     % applySineWindow Apply Sine window to NMR time-domain data
%     % 
%     % Usage:
%     %   windowed_data = applySineWindow(data, off, endVal, pow, c)
%     %
%     % Inputs:
%     %   data  - Input time-domain data (vector)
%     %   off   - Offset value (start of the window function)
%     %   endVal - End value (end of the window function)
%     %   pow   - Power (exponent) of the window function
%     %   c     - First-point scale factor
%     %
%     % Output:
%     %   windowed_data - Data after applying the Sine window function
% 
%     % Number of data points
%     N = length(data)*2;
% 
%     % Generate the Sine window function
%     window = sin(pi * ((0:N-1)' / (N-1)).^pow);
%     % Apply the first-point scale factor
%     window(1) = window(1) * c;
% 
%     % Apply the window function to the data
%     windowed_data = data .* window';
% end
function windowed_data = applySineWindow(data, off, endVal, pow, c)
    
    N = length(data);

    % Generate the Sine window function
    t = linspace(off, endVal, N);
    window = sin(pi * t.^pow);
%     figure,plot(real(window))
    % Apply the first-point scale factor
    window(1) = window(1) * c;

    % Apply the window function to the data
    windowed_data = data.* window;
  
end