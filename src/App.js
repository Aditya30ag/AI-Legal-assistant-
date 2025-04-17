import './App.css';
import { createBrowserRouter,RouterProvider} from "react-router-dom";
import LegalAssistantApp from './component/LegalAssistantApp';

function App() {
  
 
  const router=createBrowserRouter([
    {
      path:"/",
      element:<><LegalAssistantApp/></>
    },
    
  ])
 
  return (
    <>
    <RouterProvider router={router}/>
    </>
  );
}

export default App;
